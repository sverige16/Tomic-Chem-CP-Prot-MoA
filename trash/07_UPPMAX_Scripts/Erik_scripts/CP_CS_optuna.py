from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem


#from pyDeepInsight import ImageTransformer, Norm2Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import sys
import optuna 
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')

from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    create_splits, 
    choose_device,
    checking_veracity_of_data, 
    dict_splitting_into_tensor, 
    extract_tprofile, 
    EarlyStopper, 
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    conf_matrix_and_class_report,
    tprofiles_gc_too_func, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    Transcriptomic_Profiles_gc_too, 
    Transcriptomic_Profiles_numpy,
    set_bool_hqdose, 
    set_bool_npy, 
    FocalLoss, 
    np_array_transform,
    apply_class_weights_GE, 
    adapt_training_loop, 
    adapt_validation_loop, 
    adapt_test_loop,
    channel_5_numpy,
    splitting,
    cmpd_id_overlap_check, 
    inputs_equalto_labels_check,
    pre_processing,
    set_parameter_requires_grad,
    smiles_to_array,
    extracting_pretrained_single_models,
    accessing_all_folds_csv,
    CP_driving_code,
    returning_smile_string,
    dubbelcheck_dataset_length
)
from Helper_Models import (image_network, MyRotationTransform, Chemical_Structure_Model)
from efficientnet_pytorch import EfficientNet 



#!/usr/bin/env python
# coding: utf-8

# !pip install rdkit-pypi

# import statements
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import re



import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score


class CS_CP_Model(nn.Module):
    def __init__(self, modelCP, modelCS, init_nodes):
        super(CS_CP_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer1 = nn.Linear(int(1280 + 103), init_nodes)
        self.selu = nn.SELU()
        #self.Dropout = nn.Dropout(p = 0.3)
        #self.linear_layer2 = nn.Linear(128,10)
       
    def forward(self, x1in, x2in):
        x1 = self.modelCP(x1in)
        x2 = self.modelCS(x2in)
        x  = torch.cat((torch.squeeze(x1), x2), dim = 1)
        #x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.selu(self.linear_layer1(x))
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
        self.image = images_np

    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data)
    
    def __getitem__(self,idx):
        '''Retrieving the compound'''
        meta_data_idx = self.meta_data.index[idx]  
        label = self.meta_data["moa"][meta_data_idx]
        cmpdID = self.meta_data["compound"][meta_data_idx]
        image = torch.tensor(self.image[meta_data_idx])
        smile_string = returning_smile_string(self.compound_df, cmpdID)
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return torch.squeeze(image.float()), compound_array.float(), label_tensor.float() # returns 

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
incl_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

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
batch_size = 4
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
'''
# load individual models
print("Loading Pretrained Models...")
modelCS = Chemical_Structure_Model(num_features = 2048, num_targets=   num_classes)
modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

# create a model combining both models
model = CS_CP_Model(modelCP, modelCS)
'''
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
            in_features = pretrained_model.linear_layer1.out_features
            self.num_classes = num_classes
            
            layers = []
            for i in range(n_layers):
                out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 250)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.LeakyReLU())
                layers.append(nn.BatchNorm1d(out_features))
                p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))
                in_features = out_features
            layers.append(nn.Linear(out_features, num_classes))

            # Additional layers for feature extraction
            self.additional_layers = nn.Sequential(*layers)

        def forward(self, x, y):
            x = self.base_model(x, y)
            x = self.additional_layers(x)
            return x
    
    
    # load individual models
    print("Loading Pretrained Models...")
    modelCS = Chemical_Structure_Model(num_features = 2048, num_targets=   num_classes)
    modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
    modelCP = image_network()
    modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

    # create a model combining both models
    #model = CS_CP_Model(modelCP, modelCS)
    init_nodes = trial.suggest_int('init_nodes', 4, 250)
    pretrained_model = CS_CP_Model(modelCP, modelCS, init_nodes)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    model = Extended_Model(trial, pretrained_model, num_classes, n_layers = n_layers) 

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

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        
    
    fe_ft = 'fe'
    if fe_ft == 'fe':
        patience = 0
        max_epochs = trial.suggest_int('max_epochs_fe', 5, 25)
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
    lr_ft = trial.suggest_float('lr_ft', 1e-6, 1e-1, log=True)
    optimizer_ft = getattr(torch.optim, optimizer_name_ft)(model.parameters(), lr=lr_ft)
    #scheduler_name_ft = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    scheduler_ft = 'false'
    '''
    if scheduler_name_ft == "StepLR":
        scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name_ft == "ExponentialLR":
        scheduler_ft = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name_ft == "CosineAnnealingLR":
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))
    '''

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

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        
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
                            df_train_labels = training_meta_data),
                            n_trials = 80)


print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/data_for_models/" + model_name + '_' + fe_ft + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()


