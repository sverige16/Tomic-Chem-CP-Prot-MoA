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

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import smiles_to_array,  accessing_correct_fold_csv_files, create_splits, dict_splitting_into_tensor, splitting
from Erik_alll_helper_functions import EarlyStopper,  val_vs_train_loss, val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import  create_terminal_table, upload_to_neptune
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
    inputs_equalto_labels_check
)

from Helper_Models import (Chemical_Structure_Model)

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
max_epochs = 1000 # number of epochs we are going to run 
#apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()
model_name = "Chemical_Structure"

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

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
        smile_string = self.compound_df["SMILES"][idx]      # returns smiles by using compound as keys
        compound_array = smiles_to_array(smile_string)
        label = self.compound_labels.iloc[idx]             # extract classification using index
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return compound_array.float(), label_tensor.float() # returns the image and the correct label

#----------------------------------------- model -----------------------------------------#
# donwload compound list for both v1 and v2
compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")

file_name = "erik10_hq_8_12"
for fold_int in range(1,5):
    print(f'Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)
    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    # download dictionary which associates moa with a tensor

    dict_moa = dict_splitting_into_tensor(training_set)
    assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique())

    test_data_lst = list(test_set['Compound_ID'])
    train_data_lst = list(training_set['Compound_ID'])
    valid_data_lst = list(validation_set['Compound_ID'])

    cmpd_id_overlap_check(train_data_lst, valid_data_lst, test_data_lst)

    num_classes = len(training_set.moa.unique())


    # split data into labels and inputs
    training_df, train_labels = splitting(training_set)
    validation_df, validation_labels = splitting(validation_set)
    test_df, test_labels = splitting(test_set)



    batch_size = 200
    # parameters
    params = {'batch_size' : batch_size,
            'num_workers' : 3,
            'shuffle' : True,
            'prefetch_factor' : 2} 


    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    #device = torch.device('cpu')
    print(f'Training on device {device}. ' )


    # Create datasets with relevant data and labels
    training_dataset = Dataset(training_df, train_labels, dict_moa)
    valid_dataset = Dataset(validation_df, validation_labels, dict_moa)
    test_dataset = Dataset(test_df, test_labels, dict_moa)
    inputs_equalto_labels_check(training_df, train_labels, validation_df, validation_labels, test_df, test_labels)

    # create generator that randomly takes indices from the training set
    training_generator = torch.utils.data.DataLoader(training_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)


    # Creating Archi

    learning_rate = 0.07025377271459003
    model =  Chemical_Structure_Model(num_features = 2048,
                                        num_targets = num_classes)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.7390193870363918)

    yn_class_weights = True
    if yn_class_weights:
        class_weights = apply_class_weights_CL(training_set, dict_moa, device)
    # choosing loss_function 
    loss_fn_str = 'cross'
    loss_fn_train_str = 'ols'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                    loss_fn_train_str = loss_fn_train_str, 
                                                    class_weights=class_weights,
                                                    alpha = 0.8945292907426503,
                                                    smoothing =0.25978341135337146)

    #----------------------------------------------------- Training and validation ----------------------------------#

    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
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
                early_patience = 50)

    #----------------------------------------- Assessing model on test data -----------------------------------------#
    model_test = Chemical_Structure_Model(num_features = 2048,
                                        num_targets = num_classes)
    model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt")['model_state_dict'])
        #----------------------------------------- Assessing model on test data -----------------------------------------#
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model_test,
                                            test_loader = test_generator,
                                            device = device)


    #-------------------------------- Writing interesting info into terminal ------------------------# 

    val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images') 
    val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now, model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images')

    #-------------------------------- Writing interesting info into terminal ------------------------# 

    end = time.time()

    elapsed_time = program_elapsed_time(start, end)

    create_terminal_table(elapsed_time, all_labels, all_predictions)
    upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = "False",
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
                        pixel_size = 0,
                        loss_fn_train = loss_fn_train)
