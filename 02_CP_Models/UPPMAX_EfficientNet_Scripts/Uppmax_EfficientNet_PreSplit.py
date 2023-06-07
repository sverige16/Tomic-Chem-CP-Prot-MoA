#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import numpy as np
import datetime
import time
from tabulate import tabulate

# Torch
import torch
import neptune.new as neptune
import ast

from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    create_splits, 
    choose_device,
    dict_splitting_into_tensor, 
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    set_bool_hqdose, 
    adapt_training_loop,  
    adapt_test_loop,
    cmpd_id_overlap_check, 
    adapt_training_loop,
    accessing_all_folds_csv
)
from Helper_Models import (image_network)
# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparametersd
max_epochs = 100 # number of epochs we are going to run 
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
for fold_int in range(1,5):
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

    # checking to see that the unique moa classes are identical across training, validation and test set
    assert set(training_df.label.unique()) == set(validation_df.label.unique()) == set(test_df.label.unique())
    num_classes = len(training_set.moa.unique())

    # showing that I have no GPUs
    world_size = torch.cuda.device_count()
    # print(world_size)

    # importing data normalization pandas dataframe
    pd_image_norm = pd.read_csv('/home/jovyan/data_for_models/dmso_stats_v1v2.csv')


    batch_size = 8
    # parameters
    params = {'batch_size' : batch_size,
            'num_workers' : 2,
            'shuffle' : True,
            'prefetch_factor' : 1} 
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

    # optimizer_algorithm
    learning_rate = 0.0028317822185092425
    yn_class_weights = True
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_CL(training_set, dict_moa, device)
    else:
        class_weights = None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma =  0.8643905296307451 )
    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'false'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                        loss_fn_train_str = loss_fn_train_str,
                                                        class_weights = class_weights)
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
                early_patience = 20)
    #--------------------------------- Assessing model on test data ------------------------------#
    model_test = image_network()
    model_test.load_state_dict(torch.load('/home/jovyan/saved_models/' + model_name + ".pt")['model_state_dict'])
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model_test,
                                            test_loader = test_generator,
                                            device = device)
        

    #---------------------------------------- Visual Assessment ---------------------------------# 

    val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name, '/home/jovyan/saved_images') 
    val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/saved_images')


    #-------------------------------- Writing interesting info into terminal ------------------------# 

    end = time.time()

    elapsed_time = program_elapsed_time(start, end)

    create_terminal_table(elapsed_time, all_labels, all_predictions)
    upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = "mean and std",
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
                        loss_fn_train = loss_fn_train,
                        pixel_size = 256 
                    )

