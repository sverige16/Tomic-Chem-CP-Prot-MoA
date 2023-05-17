

#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate
import albumentations as A
import random
import optuna

# Torch
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn

import seaborn as sns
import neptune.new as neptune

# Image analysis packages
import albumentations as A 
import cv2           
#pip install --upgrade efficientnet-pytorch  
from efficientnet_pytorch import EfficientNet
import re     
import ast




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
    channel_5_numpy,
    splitting,
    cmpd_id_overlap_check, 
    inputs_equalto_labels_check,
    adapt_training_loop,
    accessing_all_folds_csv
)
from Helper_Models import (image_network, MyRotationTransform)
# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparametersd
max_epochs = 50 # number of epochs we are going to run 
using_cuda = True # to use available GPUs
model_name = "Cell_Painting"

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

class UPPMAX_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, pre_fix_meta_data, pre_fix_image_df):
        self.image_df = pre_fix_image_df
        self.meta_data = pre_fix_meta_data
        self.index_list = pre_fix_meta_data.index.tolist()
    def __len__(self):
        ''' The number of data points '''
        return len(self.meta_data)

    def __getitem__(self, idx):
        '''Retreiving the image '''
        actual_idx = self.index_list[idx]                 # extract index using idx
        image = torch.tensor(self.image_df[actual_idx])           # extract path using index
        label_str = self.meta_data["label"][actual_idx]          # extract calssification using index
        data_str = label_str.replace('tensor(', '').replace(', dtype=torch.float64)', '')
        data_tuple = ast.literal_eval(data_str)
        tensor_label = torch.tensor(data_tuple, dtype=torch.float64)
        return image.float(), tensor_label



paths_v1v2 = pd.read_csv('/home/jovyan/data_for_models/uppmax_image.csv')
images = np.load('/home/jovyan/data_for_models/uppmax_image.npy', allow_pickle=True)
file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
for fold_int in range(0,1):
    print(f'Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    # download dictionary which associates moa with a number
    dict_moa = dict_splitting_into_tensor(training_set)
    assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique())

    # extract compound IDs
    test_data_lst= list(test_set["Compound_ID"].unique())
    train_data_lst= list(training_set["Compound_ID"].unique())
    valid_data_lst= list(validation_set["Compound_ID"].unique())

    # check to make sure the compound IDs do not overlapp between the training, validation and test sets
    cmpd_id_overlap_check(train_data_lst, valid_data_lst, test_data_lst)

    # extracting all the paths to the images where we have a Compound in the respective lists
    training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)]
    validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)]
    test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)]

    # removing the compounds that have a moa class that contains a "|" as this is not supported by the model
    #training_df = training_df[training_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
    #validation_df = validation_df[validation_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
    #test_df = test_df[test_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)

    # Checking to see if the compounds after removing from paths_v1v2 are the same as the ones in the training, validation and test sets
    # no loss should occur, but it does occur
    #assert len(list(training_df.compound.unique())) == len(train_data_lst)
    #assert len(list(validation_df.compound.unique())) == len(valid_data_lst)
    #assert len(list(test_df.compound.unique())) == len(test_data_lst)

    # checking to see that the unique moa classes are identical across training, validation and test set
    assert set(training_df.label.unique()) == set(validation_df.label.unique()) == set(test_df.label.unique())
    num_classes = len(training_set.moa.unique())

    '''
    # split data into labels and inputs
    training_df, train_labels = splitting(training_df)
    validation_df, validation_labels = splitting(validation_df)
    test_df, test_labels = splitting(test_df)
    '''
    # showing that I have no GPUs
    world_size = torch.cuda.device_count()
    # print(world_size)

    # importing data normalization pandas dataframe
    pd_image_norm = pd.read_csv('/home/jovyan/data_for_models/dmso_stats_v1v2.csv')

    import multiprocessing

    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Number of available CPU cores: {num_cpu_cores}")

    batch_size = 8
    # parameters
    params = {'batch_size' : batch_size,
            'num_workers' : 2,
            'shuffle' : True,
            'prefetch_factor': 1} 
            #'collate_fn' : custom_collate_fn } 
            

    device = choose_device(using_cuda = True)

    # Create datasets with relevant data and labels
    training_dataset = UPPMAX_Image_Dataset(training_df, images)
    valid_dataset = UPPMAX_Image_Dataset(validation_df, images)
    test_dataset = UPPMAX_Image_Dataset(test_df, images)

    # create generator that randomly takes indices from the training set
    training_generator = torch.utils.data.DataLoader(training_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)

    #-------------------------- Creating MLP Architecture ------------------------------------------#
    model = image_network()


    #------------------------ Class weights, optimizer, and loss function ---------------------------------#

    '''    
    # optimizer_algorithm
    learning_rate = 0.001
    yn_class_weights = True
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_CL(training_set, dict_moa, device)
    else:
        class_weights = None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)
    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'false'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                        loss_fn_train_str = loss_fn_train_str,
                                               class_weights = class_weights)
    '''
    #------------------------------   Calling functions --------------------------- #

    
    def objectiv(trial, num_feat, num_classes, training_generator, validation_generator, testing_generator):
    
                
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
        loss_fn_str = trial.suggest_categorical('loss_fn', [ "cross", 'BCE'])
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
            

        max_epochs = 60

        
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
                    early_patience = 10)


        #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
        return max(val_f1_score_per_epoch)

    storage = 'sqlite:///' + model_name + '_3' '.db'
    study = optuna.create_study(direction='maximize',
                                storage = storage)
    study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                        num_classes = 10, 
                                        training_generator= training_generator, 
                                        validation_generator = validation_generator,
                                        testing_generator = test_generator), 
                                        n_trials=65)
    print("Number of finished trials: {}".format(len(study.trials)))
    print(study.best_params)
    print(study.best_value)

    f = open("/home/" + model_name + '_' + now +'_best_params.txt',"w")
    # write file
    f.write(model_name)
    f.write("Best Parameters: " + str(study.best_params))
    f.write("Best Value: " + str(study.best_value))
    # close file
    f.close()
