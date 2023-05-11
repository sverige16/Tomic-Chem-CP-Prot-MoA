
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
    channel_5_numpy,
    splitting,
    cmpd_id_overlap_check, 
    inputs_equalto_labels_check,
    adapt_training_loop,
    accessing_all_folds_csv
)
from Helper_Models import (image_network, MyRotationTransform)

class UPPMAX_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_df, labels_df, dict_moa, transform=None, image_normalization=None):
        self.img_labels = labels_df
        # print(self.img_labels)
        self.paths_df = paths_df
        self.transform = transform
        self.im_norm  = image_normalization
        self.dict_moa = dict_moa

    def __len__(self):
        ''' The number of data points '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''Retreiving the image '''
        # ID = self.list_ID[idx]
        image = channel_5_numpy(self.paths_df, idx, self.im_norm) # extract image from csv using index
        label = self.img_labels[idx]          # extract calssification using index
        #label = torch.tensor(label, dtype=torch.short)
        label_tensor = torch.from_numpy(self.dict_moa[label]) # convert label to tensor
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        plate = self.paths_df["plate"][idx]
        well = self.paths_df["well"][idx]
        compound = self.paths_df["compound"][idx]
        return plate, well, compound, label_tensor, image

rotation_transform = MyRotationTransform([0,90,180,270])
# on the fly data augmentation 
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])

paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

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
    training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
    validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
    test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)

    # removing the compounds that have a moa class that contains a "|" as this is not supported by the model
    training_df = training_df[training_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
    validation_df = validation_df[validation_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
    test_df = test_df[test_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)

    # Checking to see if the compounds after removing from paths_v1v2 are the same as the ones in the training, validation and test sets
    # no loss should occur, but it does occur
    #assert len(list(training_df.compound.unique())) == len(train_data_lst)
    #assert len(list(validation_df.compound.unique())) == len(valid_data_lst)
    #assert len(list(test_df.compound.unique())) == len(test_data_lst)

    # checking to see that the unique moa classes are identical across training, validation and test set
    assert set(training_df.moa.unique()) == set(validation_df.moa.unique()) == set(test_df.moa.unique())
    num_classes = len(training_set.moa.unique())

    entire_dataset = pd.concat([training_df, validation_df, test_df]).reset_index(drop=True)
    '''
    # split data into labels and inputs
    training_df, train_labels = splitting(training_df)
    validation_df, validation_labels = splitting(validation_df)
    test_df, test_labels = splitting(test_df)
    '''
    entire_dataset, entire_labels = splitting(entire_dataset)
    # showing that I have no GPUs
    world_size = torch.cuda.device_count()
    # print(world_size)

    # importing data normalization pandas dataframe
    pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')


    batch_size = 18
    # parameters
    params = {'batch_size' : batch_size,
            'num_workers' : 3,
            'shuffle' : True,
            'prefetch_factor' : 1} 
            #'collate_fn' : custom_collate_fn } 
            

    device = choose_device(using_cuda = True)

    # Create datasets with relevant data and labels
    entire_dataset = UPPMAX_Image_Dataset(entire_dataset, entire_labels, dict_moa, transform = train_transforms, image_normalization= pd_image_norm) 
    '''
    training_dataset = UPPMAX_Image_Dataset(training_df, train_labels, dict_moa, transform = train_transforms, image_normalization= pd_image_norm)
    valid_dataset = UPPMAX_Image_Dataset(validation_df, validation_labels, dict_moa, image_normalization= pd_image_norm)
    test_dataset = UPPMAX_Image_Dataset(test_df, test_labels, dict_moa, image_normalization= pd_image_norm)

    inputs_equalto_labels_check(training_df, train_labels, validation_df, validation_labels, test_df, test_labels)
'''
    import csv

    # Define the filename for the CSV file
    storage_path = '/scratch2-shared/erikep/'
    csv_filename = "uppmax_image.csv"
    npy_filename = "uppmax_image.npy"

    # Create a CSV writer object
    csv_file = open(storage_path + csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the header row to the CSV file
    csv_writer.writerow(['plate', 'well', 'compound', 'label', 'image'])
    images = np.empty(((len(entire_dataset)), 5, 256, 256))
    # Iterate over the images in the dataset and write the data to the CSV file
    for idx in tqdm(range(len(entire_dataset)), desc="Writing to CSV file", leave  = False):
    #for idx in tqdm(range(10), desc="Writing to CSV file", leave  = False):
        # Get the input and label for this image
        plate_, well_, compound_, label_, image_ = entire_dataset[idx]

        # Convert the input and label data to Python types
        plate_ = plate_
        well_ = well_
        compound_ = compound_
        label_ = label_
        images[idx] = image_.cpu().numpy()

        # Write the row to the CSV file
        csv_writer.writerow([plate_, well_, compound_, label_])

    # Close the CSV file
    csv_file.close()

    # Save the numpy array to a .npy file
    np.save(storage_path + npy_filename, images)
            