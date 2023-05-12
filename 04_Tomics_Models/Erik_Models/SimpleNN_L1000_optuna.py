#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix
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

import torch.nn.functional as F
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,log_loss, f1_score, accuracy_score
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN
from sklearn.feature_selection import VarianceThreshold

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import time
import heapq
import optuna

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    apply_class_weights_GE,
    accessing_correct_fold_csv_files, 
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
    channel_5_numpy,
    splitting,
    cmpd_id_overlap_check, 
    inputs_equalto_labels_check,
    checking_veracity_of_data,
    check_overlap_sigid
)



using_cuda = True
hidden_size = 1000
model_name = 'SimpleNN_L1000_optuna'


# In[39]:


# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

import torch
import torch.nn as nn


class Transcriptomic_Profiles(torch.utils.data.Dataset):
    def __init__(self, gc_too, split, dict_moa):
        #self.tprofile_labels = labels
        self.profiles_gc_too = gc_too
        self.split_sets = split
        self.dict_moa = dict_moa
        
    def __len__(self):
        ''' The number of data points '''
        return len(self.split_sets)

    def __getitem__(self, idx):
        '''Retreiving the transcriptomic profile and label
        Pseudocode:
        1. Extract the transcriptomic profile using the index along with sig_id
        2. Extract the label from t_profile
        3. Use sig_id to extract the label from split_sets
        4. Convert label to one hot encoding using function
        5. Convert to torch tensors and return'''

        t_profile = extract_tprofile(self.profiles_gc_too, idx)          # extract image from csv using index
        t_sig_id = t_profile[0][0]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        t_moa = torch.tensor(self.dict_moa[moa_key])
        t_profile_features = torch.tensor(t_profile[1])       # turn t profile into a floating torch tensor
        
        return torch.squeeze(t_profile_features), t_moa 


batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 5e-3
num_feat = 0
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2} 
          
if using_cuda:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
else:
    device = torch.device('cpu')
print(f'Training on device {device}. ' )



file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
# Creating a  dictionary of the one hot encoded labels
dict_moa = dict_splitting_into_tensor(training_set)


# checking that no overlap in sig_id exists between training, test, validation sets
check_overlap_sigid(L1000_training, L1000_validation, L1000_test)
# shuffling training and validation data
L1000_training = L1000_training.sample(frac = 1, random_state = 1)
L1000_validation = L1000_validation.sample(frac = 1, random_state = 1)
L1000_test = L1000_test.sample(frac = 1, random_state = 1)

print("extracting training transcriptomes")
profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
train_np = np_array_transform(profiles_gc_too_train)
print("extracting validation transcriptomes")
profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
valid_np = np_array_transform(profiles_gc_too_valid)
print("extracting test transcriptomes")
profiles_gc_too_test = tprofiles_gc_too_func(L1000_test, clue_gene)
test_np = np_array_transform(profiles_gc_too_test)


num_classes = len(L1000_training["moa"].unique())
'''
# create generator that randomly takes indices from the training set
training_dataset = Transcriptomic_Profiles(profiles_gc_too_train, L1000_training, dict_moa)
training_generator = torch.utils.data.DataLoader(training_dataset, **params, sampler = sampler)

# create generator that randomly takes indices from the validation set
validation_dataset = Transcriptomic_Profiles(profiles_gc_too_valid, L1000_validation, dict_moa)
validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

test_dataset = Transcriptomic_Profiles(profiles_gc_too_test, L1000_test, dict_moa)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)
'''

# create generator that randomly takes indices from the training set
training_dataset = Transcriptomic_Profiles_numpy(train_np, L1000_training, dict_moa)
training_generator = torch.utils.data.DataLoader(training_dataset, **params)

# create generator that randomly takes indices from the validation set
validation_dataset = Transcriptomic_Profiles_numpy(valid_np, L1000_validation, dict_moa)
validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

test_dataset = Transcriptomic_Profiles_numpy(test_np, L1000_test, dict_moa)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)




# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
max_epochs = 250 # number of epochs we are going to run 
# apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

# --------------------------Function to perform training, validation, testing, and assessment ------------------#

#---------------------------------------- Visual Assessment ---------------------------------# 



def define_model(trial, num_feat, num_classes):
    # optimizing hidden layers, hidden units and drop out ratio
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    in_features = num_feat
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(out_features))
        p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(out_features, num_classes))
    return nn.Sequential(*layers)

def objectiv(trial, num_feat, num_classes, training_generator, validation_generator):
    # generate the model
    model = define_model(trial, num_feat, num_classes).to(device)
    
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
        class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'focal', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false','ols'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        

    max_epochs = 150

    
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
                val_str = 'f1',
                early_patience = 15)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return max(val_f1_score_per_epoch)
storage = 'sqlite:///' + model_name + '.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = num_classes, 
                                      training_generator= training_generator, 
                                      validation_generator = validation_generator), 
                                      n_trials=100)

#-------------------------------- Writing interesting info into terminal ------------------------# 

print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/" + model_name + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()