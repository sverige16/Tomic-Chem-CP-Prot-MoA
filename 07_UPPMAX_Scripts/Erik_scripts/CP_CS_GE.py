
#!/usr/bin/env python
# coding: utf-8

# !pip install rdkit-pypi
# Torch
import torch
import torch.nn as nn
import neptune.new as neptune
import optuna
import datetime
import time
import sys
import h5py


sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    choose_device,
    different_loss_functions, 
    apply_class_weights_GE, 
    adapt_training_loop, 
    GE_driving_code,
    extracting_pretrained_single_models,
    CP_driving_code,
    set_parameter_requires_grad,
    dubbelcheck_dataset_length,
    returning_smile_string,
    smiles_to_array,
    extracting_Tprofiles_with_cmpdID

    
)
from Helper_Models import ( 
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           Chemical_Structure_Model,
                           Modified_GE_Model,
)

from Helper_Models import (image_network, Chemical_Structure_Model)


model_name = "CP_CS_GE"


class CS_CP_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, meta_data_CP,  checking_mechanism, dict_moa, tprofiles_df, split_sets, dict_cell_line, images_H):
        self.compound_df = compound_df
        self.meta_data_CP = meta_data_CP
        self.dict_moa = dict_moa
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
        self.images_H = images_H
        
        
    
    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data_CP) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data_CP)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        meta_data_idx = self.meta_data_CP.index[idx]  
        label = self.meta_data_CP["moa"][meta_data_idx]
        cmpdID = self.meta_data_CP["compound"][meta_data_idx]
        with h5py.File(self.images_H, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])
        smile_string = returning_smile_string(self.compound_df, cmpdID)
        compound_array = smiles_to_array(smile_string)
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        t_profile, t_cell_line = extracting_Tprofiles_with_cmpdID(self.tprofiles_df, self.split_sets, cmpdID, self.dict_cell_line)
                 # convert label to number
        t_profile, t_cell_line = extracting_Tprofiles_with_cmpdID(self.tprofiles_df, self.split_sets, cmpdID, self.dict_cell_line)
        label_tensor = torch.from_numpy(self.dict_moa[label])   
        return compound_array.float(), image.float(), t_profile, t_cell_line, label_tensor # returns 
    
    
# create a model combining both models
class CS_CP_GE_Model(nn.Module):
    def __init__(self, modelCS, modelCP, modelGE):
        super(CS_CP_GE_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelGE = modelGE
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer1 = nn.Linear(int(103 + 20 + 1280), 25)
        self.selu = nn.SELU()
        self.Dropout = nn.Dropout(p = 0.3)
        self.linear_layer2 = nn.Linear(25,10)
       
    def forward(self, x1in, x2in, x3in, x4in):
        x1 = self.modelCS(x1in)
        x2 = self.modelCP(x2in)
        x3 = self.modelGE(x3in, x4in)
        x  = torch.cat((x1, torch.squeeze(x2), x3), dim = 1)
        x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer2(x)
        return output

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
#---------------------------------------------------------------------------------------------------------------------------------------#

# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 12
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# -----------------------------------------Prepping Ensemble Model ---------------------#


file_name = "erik10_hq_8_12"
fold_num = 0
train_np_GE, valid_np_GE, test_np_GE, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training_GE, L1000_validation_GE, L1000_test_GE = GE_driving_code(file_name, fold_num)

training_df_CP, validation_df_CP, test_df_CP, paths_v1v2_CP, df_train_labels_CP, dict_moa, pd_image_norm_CP, df_val_labels_CP, df_test_labels_CP, num_classes, cmpd_training_set, cmpd_validation_set, cmpd_test_set = CP_driving_code(file_name, fold_num)

# download csvs with all the data pre split

# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 10
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# load individual models
print("Loading Pretrained Models...")

# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CP_GE = CS_CP_GE_Dataset(cmpd_training_set, training_df_CP, ["train" , "CP", fold_num], dict_moa, train_np_GE, L1000_training_GE, dict_cell_lines, transform = train_transforms, im_norm= pd_image_norm_CP)
valid_dataset_CP_GE = CS_CP_GE_Dataset(cmpd_validation_set, validation_df_CP, ["valid" , "CP", fold_num], dict_moa, valid_np_GE, L1000_validation_GE, dict_cell_lines, im_norm= pd_image_norm_CP)
test_dataset_CP_GE = CS_CP_GE_Dataset(cmpd_test_set, test_df_CP, ["test" , "CP", fold_num], dict_moa, test_np_GE, L1000_test_GE, dict_cell_lines, im_norm= pd_image_norm_CP)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CP_GE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CP_GE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CP_GE, **params)

# create a model combining both models
cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCS = Chemical_Structure_Model(num_features = 2048, num_targets = 10)
modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

modified_GE_model = Modified_GE_Model(modelGE)
model = CS_CP_GE_Model(modelCS, modelCP, modified_GE_model)




set_parameter_requires_grad(model, feature_extracting = True, added_layers = 1)

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = True
class_weights = apply_class_weights_CL(df_train_labels_CP, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                  class_weights=class_weights)
my_lr_scheduler = 'false'
num_epochs_fe = 1
#n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false")
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = num_epochs_fe,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              loss_fn_train = loss_fn_train,
              loss_fn_str = loss_fn_str,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              my_lr_scheduler = my_lr_scheduler,
              model_name= model_name,
              device = device,
              val_str = 'fe',
              early_patience = 0)

print('Fine Tuning in Progress')
set_parameter_requires_grad(model, feature_extracting = False)
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = True
class_weights = apply_class_weights_CL(df_train_labels_CP, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                  class_weights=class_weights)
my_lr_scheduler = 'false'
num_epochs_ft = 1
#----------------------------------------------------- Training and validation ----------------------------------#

train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = num_epochs_ft,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              loss_fn_train = loss_fn_train,
              loss_fn_str = loss_fn_str,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              my_lr_scheduler = my_lr_scheduler,
              model_name=model_name,
              device = device,
              val_str = 'f1',
              early_patience = 10)
#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CP_GE_Model(modelCP, modelGE)
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model, 
                    test_loader = test_generator, 
                    device = device)
# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images')


#-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = True,
                    yn_class_weights = incl_class_weights,
                    learning_rate = learning_rate, 
                    elapsed_time = elapsed_time, 
                    num_epochs = num_epochs,
                    loss_fn = loss_fn,
                    all_labels = all_labels,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    variance_thresh = 0,
                    pixel_size = 0,
                    loss_fn_train = loss_fn_train)
