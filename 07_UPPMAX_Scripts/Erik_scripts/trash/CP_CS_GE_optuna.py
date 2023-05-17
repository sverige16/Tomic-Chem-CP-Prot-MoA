
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


# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
incl_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()
model_name = "CP_CS_GE"

class CP_GE_Dataset(torch.utils.data.Dataset):
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
class UPPMAX_CS_CP_GE_Model(nn.Module):
    def __init__(self, modelCS, modelCP, modelGE, init_nodes):
        super(UPPMAX_CS_CP_GE_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelGE = modelGE
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer_pre = nn.Linear(int(103 + 20 + 1280), init_nodes)
        self.leaky_relu = nn.LeakyReLU()
       
    def forward(self, x1in, x2in, x3in, x4in):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        x1 = self.modelCS(x1in)
        
        x2 = self.modelCP(x2in)
        x3 = self.modelGE(x3in, x4in)
        x  = torch.cat((x1, torch.squeeze(x2), x3), dim = 1)
        #output = self.leaky_relu(self.linear_layer_pre(x))
        #x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer_pre(x)
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



# -----------------------------------------Prepping Ensemble Model ---------------------#

file_name = "erik10_hq_8_12"
fold_num = 0
train_np_GE, valid_np_GE, test_np_GE, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training_GE, L1000_validation_GE, L1000_test_GE = GE_driving_code(file_name, fold_num)

training_df_CP, validation_df_CP, test_df_CP, dict_moa, cmpd_training_set, cmpd_validation_set, cmpd_test_set, images = CP_driving_code(file_name, fold_num)

# download csvs with all the data pre split
images_H = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
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
training_dataset_CP_GE = CP_GE_Dataset(compound_df = cmpd_training_set, 
                                       meta_data_CP = training_df_CP, 
                                       checking_mechanism= ["train" , "CP", fold_num],
                                       dict_moa = dict_moa, 
                                       tprofiles_df = train_np_GE, 
                                       split_sets = L1000_training_GE,
                                        dict_cell_line = dict_cell_lines, 
                                         images_H = images_H)

valid_dataset_CP_GE = CP_GE_Dataset(compound_df =cmpd_validation_set, 
                                    meta_data_CP = validation_df_CP, 
                                    checking_mechanism = ["valid" , "CP", fold_num], 
                                    dict_moa = dict_moa, 
                                    tprofiles_df = valid_np_GE, 
                                    split_sets = L1000_validation_GE, 
                                    dict_cell_line = dict_cell_lines, 
                                    images_H = images_H)
test_dataset_CP_GE = CP_GE_Dataset(compound_df = cmpd_test_set, 
                                   meta_data_CP = test_df_CP, 
                                   checking_mechanism = ["test" , "CP", fold_num], 
                                   dict_moa = dict_moa, 
                                   tprofiles_df = test_np_GE, 
                                   split_sets = L1000_test_GE, 
                                   dict_cell_line = dict_cell_lines, 
                                   images_H = images_H)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CP_GE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CP_GE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CP_GE, **params)


device = choose_device(using_cuda = True)

def optuna_combinations_fe_ft(trial, 
                            num_classes, 
                            training_generator,
                            validation_generator, 
                           
                            model_name, 
                            device, 
                            driver, 
                            fe_ft,
                            dict_moa,
                            df_train_labels = None,
                            train_np = None,
                            L1000_training = None):
    class Extended_Model(nn.Module):
        def __init__(self, trial, pretrained_model, num_classes, n_layers):
            super(Extended_Model, self).__init__()
            self.base_model = pretrained_model
            in_features = pretrained_model.linear_layer_pre.out_features
            self.num_classes = num_classes
            
            layers = []
            for i in range(n_layers):
                out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 250)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.LeakyReLU())
                #layers.append(nn.BatchNorm1d(out_features))
                p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))
                in_features = out_features
            layers.append(nn.Linear(out_features, num_classes))

            # Additional layCP_GE_Model(modelCP, modified_GE_modelers for feature extraction
            self.additional_layers = nn.Sequential(*layers)

        def forward(self, x, y, z, w):
            x = self.base_model(x, y, z, w )
            x = self.additional_layers(x)
            return x
    
    
    # load individual models
    print("Loading Pretrained Models...")
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
    #model = CS_CP_GE_Model(modelCS, modelCP, modified_GE_model)


    # create a model combining both models
    #model = CS_CP_Model(modelCP, modelCS)
    init_nodes = trial.suggest_int('init_nodes', 4, 250)
    pretrained_model = UPPMAX_CS_CP_GE_Model(modelCS, modelCP, modified_GE_model, init_nodes = init_nodes)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    model = Extended_Model(trial, pretrained_model, num_classes, n_layers = n_layers) 
    print(model)
    set_parameter_requires_grad(model, feature_extracting = True, added_layers = n_layers)

# generate the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))


    yn_class_weights = trial.suggest_categorical('yn_class_weights', [True, False])
    if yn_class_weights:     # if we want to apply class weights
        if driver == 'CP': # we want CP first if possible, since that is our driver
            class_weights = apply_class_weights_CL(df_train_labels, dict_moa, device)

        elif driver == 'GE':
            class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
        else:
            ValueError('Driver must be CP or GE')
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    
    fe_ft = 'fe'
    if fe_ft == 'fe':
        patience = 0
        max_epochs = 10
        #max_epochs = 1
        val_str = 'fe'


#------------------------------   Calling functions --------------------------- #
        train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                    optimizer = optimizer,
                    model = model,
                    loss_fn = loss_fn,
                    loss_fn_train = loss_fn_train,
                    loss_fn_str = loss_fn_str,
                    train_loader=training_generator, 
                    valid_loader=validation_generator,
                    my_lr_scheduler = scheduler,
                    model_name=model_name,
                    device = device,
                    val_str = val_str,
                    early_patience = patience)

    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    
    set_parameter_requires_grad(model, feature_extracting = False, added_layers = n_layers)

# generate the optimizer
    optimizer_name_ft = trial.suggest_categorical('optimizer_ft', ['Adam', 'RMSprop', 'SGD'])
    lr_ft = trial.suggest_float('lr_ft', 1e-4, 5e-1)
    optimizer_ft = getattr(torch.optim, optimizer_name_ft)(model.parameters(), lr=lr_ft)
    #scheduler_name_ft = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    scheduler_ft = 'false'

    yn_class_weights_ft = trial.suggest_categorical('yn_class_weights_ft', [True, False])
    if yn_class_weights_ft:     # if we want to apply class weights
        if driver == 'CP': # we want CP first if possible, since that is our driver
            class_weights_ft = apply_class_weights_CL(df_train_labels, dict_moa, device)

        elif driver == 'GE':
            class_weights_ft = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
        else:
            ValueError('Driver must be CP or GE')
    else:
        class_weights_ft = None
    loss_fn_str_ft = trial.suggest_categorical('loss_fn_ft', ['cross', 'BCE'])
    loss_fn_train_str_ft = trial.suggest_categorical('loss_train_ft', ['false'])
    loss_fn_train_ft, loss_fn_ft = different_loss_functions(
                                                      loss_fn_str= loss_fn_str_ft,
                                                      loss_fn_train_str = loss_fn_train_str_ft,
                                                      class_weights = class_weights_ft)

    fe_ft = 'ft'
    if fe_ft == 'ft':
        patience = 10
        max_epochs_ft = 60
        val_str_ft= 'f1'


#------------------------------   Calling functions --------------------------- #
        train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs_ft,
                    optimizer = optimizer_ft,
                    model = model,
                    loss_fn = loss_fn_ft,
                    loss_fn_train = loss_fn_train_ft,
                    loss_fn_str = loss_fn_str_ft,
                    train_loader=training_generator, 
                    valid_loader=validation_generator,
                    my_lr_scheduler = scheduler_ft,
                    model_name=model_name,
                    device = device,
                    val_str = val_str_ft,
                    early_patience = patience)
 

    return max(val_f1_score_per_epoch)

fe_ft = "fe"


storage = 'sqlite:///' + model_name + '_' + fe_ft +'.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: optuna_combinations_fe_ft(trial,
                            num_classes = 10, 
                            training_generator = training_generator,
                            validation_generator = validation_generator,
                        
                            model_name = model_name, 
                            device = device, 
                            driver = "CP", 
                            fe_ft = "fe",
                            dict_moa = dict_moa,
                            df_train_labels = training_df_CP),
                            n_trials = 80)



print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/" + model_name + '_' + fe_ft + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()