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
    one_input_training_loop, 
    one_input_validation_loop, 
    one_input_test_loop,
    channel_5_numpy,
    splitting
)

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
max_epochs = 1000 # number of epochs we are going to run 
using_cuda = True # to use available GPUs
model_name = "Cell_Painting"

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

class Dataset(torch.utils.data.Dataset):
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
        #return image.float(), label.long()
        return image.float(), label_tensor.float()  
class image_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b0', include_top=False, in_channels = 5)
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.Linear_last = nn.Linear(1280, num_classes) #41720
        # self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.dropout_1(self.base_model(x))
        out = out.view(-1, 1280)
        out = self.Linear_last(out)
        # out = self.softmax(out) # don't need softmax when using CrossEntropyLoss
        return out
class MyRotationTransform:
    " Rotate by one of the given angles"
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        angle = random.choice(self.angle)
        return TF.rotate(image, angle)
rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation 
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])

paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

file_name = "erik10_all"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
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
inter1 = set(test_data_lst) & set(train_data_lst)
inter2 = set(test_data_lst) & set(valid_data_lst)
inter3 = set(train_data_lst) & set(valid_data_lst)
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")

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


# split data into labels and inputs
training_df, train_labels = splitting(training_df)
validation_df, validation_labels = splitting(validation_df)
test_df, test_labels = splitting(test_df)

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
          

device = choose_device(using_cuda = True)

# Create datasets with relevant data and labels
training_dataset = Dataset(training_df, train_labels, dict_moa, transform = train_transforms, image_normalization= pd_image_norm)
valid_dataset = Dataset(validation_df, validation_labels, dict_moa, image_normalization= pd_image_norm)
test_dataset = Dataset(test_df, test_labels, dict_moa, image_normalization= pd_image_norm)

# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)

#-------------------------- Creating MLP Architecture ------------------------------------------#
model = image_network()


yn_class_weights = True
class_weights = apply_class_weights_CL(training_set, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, class_weights=class_weights)

#------------------------ Class weights, optimizer, and loss function ---------------------------------#


# optimizer_algorithm
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                        milestones=[8, 16, 22, 28, 32, 36], # List of epoch indices
                        gamma = 0.5)

#------------------------------   Calling functions --------------------------- #
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = one_input_training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              loss_fn_train = loss_fn_train,
              loss_fn_str = loss_fn_str,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              my_lr_scheduler = my_lr_scheduler,
              model_name=model_name,
              device = device)
#--------------------------------- Assessing model on test data ------------------------------#
model_test = image_network()
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = model_test,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#---------------------------------------- Visual Assessment ---------------------------------# 

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images')


#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = "mean and std and random crop",
                    yn_class_weights = 'CL',
                    learning_rate = learning_rate, 
                    elapsed_time = elapsed_time, 
                    num_epochs = num_epochs,
                    loss_fn = loss_function,
                    all_labels = all_labels,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    loss_fn_train = loss_fn_train,
                    pixel_size = 256 
                )

