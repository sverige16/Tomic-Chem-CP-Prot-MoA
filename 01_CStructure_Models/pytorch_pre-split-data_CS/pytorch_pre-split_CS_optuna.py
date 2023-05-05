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
import optuna
import neptune.new as neptune

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import smiles_to_array,  accessing_correct_fold_csv_files, create_splits, dict_splitting_into_tensor, splitting
from Erik_alll_helper_functions import EarlyStopper,  val_vs_train_loss, val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import create_terminal_table, upload_to_neptune
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
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
    check_overlap_sigid,
    accessing_all_folds_csv,
    checking_veracity_of_data,
    cmpd_id_overlap_check,
    inputs_equalto_labels_check,
    splitting
)

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 250 # number of epochs we are going to run 
apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'Chemical_Structure_Optuna'

#---------------------------------------------------------------------------------------------------------------------------------------#
# create Torch.dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, labels_df, dict_moa, transform=None):
        self.compound_labels = labels_df    # the entire length of the correct classes that we are trying to predict
        # print(self.img_labels)
        self.compound_df = compound_df        # list of indexes that are a part of training, validation, tes sets
        self.transform = transform       # any transformations done
        self.dict_moa = dict_moa

    def __len__(self):
        ''' The number of data points'''
        return len(self.compound_labels)      

    def __getitem__(self, idx):
        '''Retrieving the compound '''
        #print(idx)
        smile_string = self.compound_df["SMILES"][idx]      # returns smiles by using compound as keys
        #print(smile_string)
        compound_array = smiles_to_array(smile_string)
        #print(f' return from function: {compound}')
        #print(f' matrix: {compound_array}')
        label = self.compound_labels.iloc[idx]             # extract classification using index
        #print(f' label: {label}')
        #label = torch.tensor(label, dtype=torch.float)
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return compound_array.float(), label_tensor.float() # returns the image and the correct label

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# donwload compound list for both v1 and v2
compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")

file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)
checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
# download dictionary which associates moa with a tensor

dict_moa = dict_splitting_into_tensor(training_set)
assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique())


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_set)
validation_df, validation_labels = splitting(validation_set)
test_df, test_labels = splitting(test_set)

dict_moa = dict_splitting_into_tensor(training_set)

batch_size = 50
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )


# Create datasets with relevant data and labels
training_dataset = Dataset(training_df, train_labels, dict_moa)
valid_dataset = Dataset(validation_df, validation_labels, dict_moa)
test_dataset = Dataset(test_df, test_labels, dict_moa)

# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)

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
    def extract_three_highest_values(input_list):
        if len(input_list) < 3:
            raise ValueError("Input list must contain at least 3 elements")

        top_values = sorted(input_list, reverse=True)[:3]
        return sum(top_values)/3
    
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
        class_weights = apply_class_weights_CL(training_set, dict_moa, device)
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
                early_patience = 30)
    avg_high = extract_three_highest_values(val_f1_score_per_epoch)
    return avg_high
storage = 'sqlite:///' + model_name + '_avg''.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: objectiv(trial, num_feat = 2048, 
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
