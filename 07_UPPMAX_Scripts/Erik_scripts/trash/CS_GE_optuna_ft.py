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
import warnings
warnings.filterwarnings('ignore')

import optuna

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
import re
import warnings


# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import sys

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL,  
    create_splits, 
    choose_device,
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
    checking_veracity_of_data,
    check_overlap_sigid,
    extract_all_cell_lines,
    accessing_all_folds_csv,
    smiles_to_array,
    GE_driving_code,
    extracting_pretrained_single_models,
    set_parameter_requires_grad,
    returning_smile_string,
    dubbelcheck_dataset_length
)

from Helper_Models import (SimpleNN_Model, 
                           Transcriptomic_Profiles_Cell_Lines, 
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           image_network,
                           Chemical_Structure_Model,
                           Modified_GE_Model)
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter('error')

class CS_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, tprofiles_df, split_sets, dict_moa, dict_cell_line, checking_mechanism):
        self.compound_df = compound_df
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_moa = dict_moa
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
    def __len__(self):
        check_criteria = self.check
        assert len(self.tprofiles_df) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.tprofiles_df)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        t_profile = self.tprofiles_df.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.tprofiles_df.iloc[idx, -1]
        CID = self.split_sets["Compound ID"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        cell_line_key = self.split_sets["cell_iname"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        CID = CID.iloc[0]
        cell_line_key = cell_line_key.iloc[0]
        t_cell_line = torch.tensor(self.dict_cell_line[cell_line_key])
        smile_string = returning_smile_string(self.compound_df,CID)
        compound_array = smiles_to_array(smile_string)
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        label_tensor = torch.from_numpy(self.dict_moa[moa_key])                  # convert label to number
        t_profile_features = torch.tensor(t_profile) 
        return compound_array.float(), t_profile_features, t_cell_line.float(), label_tensor.float() # returns 
        
class CS_GE_Model(nn.Module):
    def __init__(self, modelCS, modelGE):
        super(CS_GE_Model, self).__init__()
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.modelGE = modelGE
        self.linear_layer_pre = nn.Linear(int(103 + 20), 238)
        #self.selu = nn.SELU()
        #self.Dropout_pre = nn.Dropout(p = 0.3)
        self.linear_layer0 = nn.Linear(238,151)
        self.Dropout0 = nn.Dropout(p = 0.35955137501880535)
        self.output_layer = nn.Linear(151, 10)
    
    def forward(self, x1in, x2in, x3in):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        if x1in.shape[1] != 2048:
            raise ValueError("The input shape for the compound is not correct")
        x1 = self.modelCS(x1in)
        x2 = self.modelGE(x2in, x3in)
        x  = torch.cat((x1, x2), dim = 1)
        x = self.linear_layer_pre(x)
        #x = self.Dropout_pre(self.selu(self.linear_layer_pre(x)))
        x = self.Dropout0(self.selu(self.linear_layer0(x)))
        output = self.output_layer(x)
        return output


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
model_name = 'CS_GE'
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# Downloading all relevant data frames and csv files ----------------------------------------------------------
file_name = "erik10_hq_8_12"
fold_num = 0
train_np, valid_np, test_np, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training, L1000_validation, L1000_test = GE_driving_code(file_name, fold_num)


device = choose_device(using_cuda)
# load individual models
print("Loading Pretrained Models...")
cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)


 

modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCS = Chemical_Structure_Model(num_features = 2048, num_targets = 10)
modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modified_GE_model = Modified_GE_Model(modelGE)
model = CS_GE_Model(modelCS, modified_GE_model)
    


batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 0.0000195589 
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2} 
          
using_cuda = True 
device = choose_device(using_cuda)  
# -----------------------------------------Prepping Ensemble Model ---------------------#

# Create datasets with relevant data and labels
training_dataset_CSGE = CS_GE_Dataset(training_set_cmpds, train_np, L1000_training, dict_moa, dict_cell_lines, ["train" , "GE", fold_num])
valid_dataset_CSGE = CS_GE_Dataset(validation_set_cmpds, valid_np, L1000_validation, dict_moa, dict_cell_lines, ["valid" , "GE", fold_num])
test_dataset_CSGE = CS_GE_Dataset(test_set_cmpds, test_np, L1000_test, dict_moa, dict_cell_lines, ["test" , "GE", fold_num])

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSGE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSGE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSGE, **params)



#set_parameter_requires_grad(model, feature_extracting = True, added_layers = 2)
#{'dropout_l0': 0.35955137501880535, 'init_nodes': 238,  'n_layers': 1, 'n_units_l0': 151, 
#0.47826662851170987
#{'dropout_l0': 0.35955137501880535, 'gamma': 0.8109702216604067, 'init_nodes': 238, 'loss_fn': 'cross', 'loss_train_fn': 'false', 'lr': 0.7782334038760198, 'max_epochs': 27, 'n_layers': 1, 'n_units_l0': 151, 'optimizer': 'Adam', 'scheduler': 'StepLR', 'step_size': 10, 'yn_class_weights': False}
#0.47826662851170987
set_parameter_requires_grad(model, feature_extracting = True, added_layers = 2)

learning_rate = 0.7782334038760198
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = False
if yn_class_weights == True:
   class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
else:
   class_weights = None
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train_str = 'false'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                    loss_fn_train_str = loss_fn_train_str,
                                                    
                                                  class_weights=class_weights)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8109702216604067)
num_epochs_fe = 27
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
              val_str = 'fe',
              early_patience = 0)


set_parameter_requires_grad(model, feature_extracting = False)


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
        
    

    if fe_ft == 'fe':
        patience = 0
        max_epochs = trial.suggest_int('max_epochs', 5, 50)
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
        return val_f1_score_per_epoch[-1]
    elif fe_ft == 'ft':
        patience = 20
        max_epochs = 100
        val_str = 'f1'
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

        return max(val_f1_score_per_epoch)

fe_ft = "ft"

storage = 'sqlite:///' + model_name + '_' + fe_ft +'.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: optuna_combinations_fe_ft(trial,
                            num_classes = num_classes, 
                            training_generator = training_generator,
                            validation_generator = validation_generator,
                        
                            model_name = model_name, 
                            device = device, 
                            driver = "GE", 
                            fe_ft = fe_ft,
                            dict_moa = dict_moa,
                            df_train_labels = None,
                            train_np = train_np,
                            L1000_training = L1000_training),
                            n_trials = 150)


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
