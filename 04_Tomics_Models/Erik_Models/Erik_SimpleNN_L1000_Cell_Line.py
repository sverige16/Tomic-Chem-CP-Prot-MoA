#!/usr/bin/env python
# coding: utf-8

# In[36]:


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
import re
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
    accessing_all_folds_csv
)

from Helper_Models import (SimpleNN_Model, 
                           Transcriptomic_Profiles_Cell_Lines, 
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model)



using_cuda = True
hidden_size = 1000


# In[39]:

model_name = 'SimpleNN_Cell_Line'

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")



batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 0.0005226706526289529
num_feat = 0
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2} 
using_cuda = True 
device = choose_device(using_cuda)         
file_name = "erik10_hq_8_12"
for fold_int in range(0,5):
    print(f' Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    #file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
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


    dict_cell_lines = extract_all_cell_lines(pd.concat([L1000_training, L1000_validation, L1000_test]))
    num_cell_lines = len(dict_cell_lines)



    # In[49]:



    # create generator that randomly takes indices from the training set
    training_dataset = Transcriptomic_Profiles_Cell_Lines(profiles_gc_too_train, 
                                                        L1000_training, 
                                                        dict_moa, 
                                                        dict_cell_lines)
    training_generator = torch.utils.data.DataLoader(training_dataset, **params)

    # create generator that randomly takes indices from the validation set
    validation_dataset = Transcriptomic_Profiles_Cell_Lines(profiles_gc_too_valid, 
                                                            L1000_validation, 
                                                            dict_moa,
                                                            dict_cell_lines)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

    test_dataset = Transcriptomic_Profiles_Cell_Lines(profiles_gc_too_test, 
                                                    L1000_test, 
                                                    dict_moa,
                                                    dict_cell_lines)
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
    '''


    cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
    simple_model = SimpleNN_Model(num_features = 978, num_targets = 15).to(device)

    model = Tomics_and_Cell_Line_Model(simple_model, cell_line_model, num_targets = 10)




    # In[66]:
    yn_class_weights = True
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
    else:
        class_weights = None
    optimizer = torch.optim.RMSprop(model.parameters(),  weight_decay=WEIGHT_DECAY, lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,  T_max=10)

    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'ols'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                    loss_fn_train_str = loss_fn_train_str, 
                                                    class_weights=class_weights,
                                                    smoothing = 0.24296117253545618,
                                                    alpha = 0.20730849127124368 )

    # In[67]:


    # ----------------------------------------- hyperparameters ---------------------------------------#
    # Hyperparameters
    max_epochs = 100 # number of epochs we are going to run 
    # apply_class_weights = True # weight the classes based on number of compounds
    using_cuda = True # to use available GPUs

    #----------------------------------------- pre-processing -----------------------------------------#
    start = time.time()
    now = datetime.datetime.now()
    now = now.strftime("%d_%m_%Y-%H:%M:%S")
    print("Begin Training")

    # --------------------------Function to perform training, validation, testing, and assessment ------------------


    #------------------------------   Calling functions --------------------------- #
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


    val_vs_train_loss_path = val_vs_train_loss(num_epochs = num_epochs,
                                            train_loss = train_loss_per_epoch,
                                            val_loss = val_loss_per_epoch, 
                                            now = now, 
                                            model_name = model_name, 
                                            file_name = file_name, 
                                            loss_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images') 
    val_vs_train_acc_path = val_vs_train_accuracy(num_epochs = num_epochs, 
                                                train_acc = train_acc_per_epoch, 
                                                val_acc = val_acc_per_epoch, 
                                                now = now,  
                                                model_name = model_name, 
                                                file_name = file_name, 
                                                acc_path_to_save ='/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images')


    #----------------------------------------- Assessing model on test data -----------------------------------------#
    model_test = Tomics_and_Cell_Line_Model(simple_model, cell_line_model, num_targets = num_classes)  
    model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')['model_state_dict'])
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model_test,
                                            test_loader = test_generator,
                                            device = device)


    #-------------------------------- Writing interesting info into terminal ------------------------# 
    end = time.time()
    elapsed_time = program_elapsed_time(start, end)

    create_terminal_table(elapsed_time, all_labels, all_predictions)
    upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = 'False',
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
                        learning_rate_scheduler = my_lr_scheduler,
                        loss_fn_train = loss_fn_train)