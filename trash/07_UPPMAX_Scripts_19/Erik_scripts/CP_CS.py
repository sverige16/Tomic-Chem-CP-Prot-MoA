# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import datetime
import time
import h5py
# Torch
import torch
from torchvision import transforms
import torch.nn as nn
import neptune.new as neptune
import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')

from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    adapt_training_loop, 
    adapt_test_loop,
    set_parameter_requires_grad,
    smiles_to_array,
    extracting_pretrained_single_models,
    CP_driving_code,
    returning_smile_string,
    dubbelcheck_dataset_length
)
from Helper_Models import (image_network, MyRotationTransform, Chemical_Structure_Model)

class CS_CP_Model(nn.Module):
    def __init__(self, modelCP, modelCS):
        super(CS_CP_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer_pre = nn.Linear(int(1280 + 103), 60)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.batch_norm = nn.BatchNorm1d(60)
        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=60, out_features=10   , bias=True)
                            )

    def forward(self, x1, x2):
        x1 = self.modelCP(x1)
        x2 = self.modelCS(x2)
        x  = torch.cat((torch.squeeze(x1), x2), dim = 1)
        x = self.leaky_relu(self.linear_layer_pre(x))
        output = self.additional_layers(x)
        return output
    
class CS_CP_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 meta_data,  
                 checking_mechanism, 
                 dict_moa, 
                compound_df,
                images_np
                 ):
        self.meta_data = meta_data
        self.dict_moa = dict_moa
        self.check = checking_mechanism
        self.compound_df = compound_df
        self.hdf5_file = images_np

    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data)
    
    def __getitem__(self,idx):
        '''Retrieving the compound'''
        meta_data_idx = self.meta_data.index[idx]  
        label = self.meta_data["moa"][meta_data_idx]
        cmpdID = self.meta_data["compound"][meta_data_idx]
        
        with h5py.File(self.hdf5_file, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])

        smile_string = returning_smile_string(self.compound_df, cmpdID)
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return torch.squeeze(image.float()), compound_array.float(), label_tensor.float() # returns 

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_CS'
#---------------------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

file_name = 'erik10_hq_8_12'
fold_int = 0
training_meta_data, validation_meta_data, test_df_meta_data, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, images = CP_driving_code(file_name, fold_int)

rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation for CP Images
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])
images = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CSCP = CS_CP_Dataset(meta_data =training_meta_data,   
                 checking_mechanism = ["train" , "CP", fold_int], 
                 dict_moa = dict_moa, 
                compound_df = training_set_cmpds,
                images_np = images)
valid_dataset_CSCP = CS_CP_Dataset(
                 meta_data = validation_meta_data, 
                 checking_mechanism = ["valid" , "CP", fold_int],
                 dict_moa = dict_moa, 
                compound_df = validation_set_cmpds,
                images_np = images)
test_dataset_CSCP = CS_CP_Dataset(
                 meta_data = test_df_meta_data,  
                 checking_mechanism = ["test" , "CP", fold_int],
                 dict_moa = dict_moa, 
                compound_df = test_set_cmpds,
                images_np = images)

# parameters for the dataloader
batch_size = 7
params = {'batch_size' : batch_size,
         'num_workers' : 2,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSCP, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSCP, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSCP, **params)

# load individual models
print("Loading Pretrained Models...")
modelCS = Chemical_Structure_Model(num_features = 2048, num_targets=  10)
modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

# create a model combining both models
model = CS_CP_Model(modelCP, modelCS)

# --------------------------------- Training, Test, Validation, Loops --------------------------------#

set_parameter_requires_grad(model, feature_extracting = True, added_layers = 0)

learning_rate =   0.003655104554321605
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = False
if yn_class_weights:
    class_weights = apply_class_weights_CL(training_meta_data, dict_moa, device)
else:
    class_weights = None
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train_str = 'false'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                    loss_fn_train_str = loss_fn_train_str,
                                                  class_weights=class_weights)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,  gamma = 0.8230760751803327)
num_epochs_fe = 10

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
              model_name=model_name,
              device = device,
              val_str = 'f1',
              early_patience = 0)

#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CS_CP_Model(modelCP, modelCS)
model_test.load_state_dict(torch.load('/home/jovyan/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model, 
                    test_loader = test_generator, 
                    device = device)
# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/saved_images')

#-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = True,
                    yn_class_weights = yn_class_weights,
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
