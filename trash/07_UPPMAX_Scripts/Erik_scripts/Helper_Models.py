
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
import torch.nn.functional as F
import seaborn as sns
import neptune.new as neptune

# Image analysis packages
import albumentations as A 
import cv2           
#pip install --upgrade efficientnet-pytorch  
from efficientnet_pytorch import EfficientNet
import re     
import math


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
    splitting
)
#---------------------------------- Chemical Model ----------------------------------#
modelCS = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(p = 0.7),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 10))

# ---------------------------------- Image Model ----------------------------------#
class image_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b0', include_top=False, in_channels = 5)
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.Linear_last = nn.Linear(1280, 10) #41720
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


# ---------------------------------- Transcriptomic Models ----------------------------------#

class CNN_Model(nn.Module):
    """
    1D-CNN Model
    For more info: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """
    def __init__(self, num_features = None, num_targets = None, hidden_size = None):
        super(CNN_Model, self).__init__()
        cha_1 = 256
        cha_2 = 512
        cha_3 = 512
        
        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(math.ceil(hidden_size/cha_1/2))
        cha_po_2 = int(math.ceil(hidden_size/cha_1/2/2)) * cha_3
        
        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2
        
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.3483945766265415)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.408144359891371)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)
        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.3718565001384708)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)
        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.2712714934244445)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)
        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.3823529290021656)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)
        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.flt = nn.Flatten()
        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.45601244515222367)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))
        
    ##commented out some of the batch_norms because the loss gradients returns nan values
    def forward(self, x):
        #x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)
        x = x.reshape(x.shape[0],self.cha_1, self.cha_1_reshape)
        #x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))
        x = self.ave_po_c1(x)
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x
        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))
        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s
        x = self.max_po_c2(x)
        x = self.flt(x)
        x = self.batch_norm3(x) 
        x = self.dropout3(x)
        x = self.dense3(x)
        return x


class SimpleNN_Model(nn.Module):
    """
    Simple 3-Layer FeedForward Neural Network
    
    For more info: https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold\
    /blob/main/final/Best%20LB/Training/3-stagenn-train.ipynb
    """
    def __init__(self, num_features = None, num_targets = None, hidden_size = None):
        super(SimpleNN_Model, self).__init__()
        #layer 1
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 49))
        self.batch_norm1 = nn.BatchNorm1d(49)
        self.dropout1 = nn.Dropout(0.24509543540660458)

        # output layer
        self.output = nn.Linear(49, num_targets)
        #self.dense4 = nn.utils.weight_norm(nn.Linear(48, num_targets))
        '''
        #self.batch_norm1 = nn.BatchNorm1d(num_features)
        #self.dropout1 = nn.Dropout(0.29323431985537163)
        #layer 1
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 64))
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.24509543540660458)
        #layer 2
        self.dense2 = nn.utils.weight_norm(nn.Linear(64, 43))
        self.batch_norm2 = nn.BatchNorm1d(43)
        self.dropout2 = nn.Dropout(0.30761037332988056)
        #layer 3
        self.dense3 = nn.Linear(43, 103)
        self.batch_norm3 = nn.BatchNorm1d(103)
        self.dropout3 = nn.Dropout( 0.31140834347665325)

        # output layer
        self.output = nn.Linear(103, num_targets)
        #self.dense4 = nn.utils.weight_norm(nn.Linear(48, num_targets))
        '''
    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)


        x = F.leaky_relu(self.output(x))
        '''
        #x = self.batch_norm1(x)
        #x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = F.leaky_relu(self.dense2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        x = F.leaky_relu(self.dense3(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        
        x = F.leaky_relu(self.output(x))
        '''
        
        return x

class Transcriptomic_Profiles_Cell_Lines(torch.utils.data.Dataset):
    def __init__(self, gc_too, split, dict_moa, dict_cell_line):
        #self.tprofile_labels = labels
        self.profiles_gc_too = gc_too
        self.split_sets = split
        self.dict_moa = dict_moa
        self.dict_cell_line = dict_cell_line
        
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
        # moa extraction
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        t_moa = torch.tensor(self.dict_moa[moa_key])
        # cell line extration
        cell_line_key = self.split_sets["cell_iname"][self.split_sets["sig_id"] == t_sig_id]
        cell_line_key = cell_line_key.iloc[0]
        t_cell_line = torch.tensor(self.dict_cell_line[cell_line_key])
        t_profile_features = torch.tensor(t_profile[1])       # turn t profile into a floating torch tensor
        
        return torch.squeeze(t_profile_features), t_cell_line.float(), t_moa 


class Transcriptomic_Profiles_gc_too(torch.utils.data.Dataset):
    '''
    Works with profiles_gc_too_func to create a dataset of transcriptomic profiles and labels
    '''
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

class Transcriptomic_Profiles_numpy(torch.utils.data.Dataset):
    '''
    Works with profiles_gc_too_func to create a dataset of transcriptomic profiles and labels
    '''
    def __init__(self, np_array, split, dict_moa):
        #self.tprofile_labels = labels
        self.profiles_np_array = np_array
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

        t_profile = self.profiles_np_array.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.profiles_np_array.iloc[idx, -1]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        t_moa = torch.tensor(self.dict_moa[moa_key])
        t_profile_features = torch.tensor(t_profile)       # turn t profile into a floating torch tensor
        
        return t_profile_features, t_moa 

class Cell_Line_Model(nn.Module):
    def __init__(self, num_features = None, num_targets = None):
        super(Cell_Line_Model, self).__init__()
        self.dense1 = nn.Linear(num_features, 20)
        self.dense2 = nn.Linear(20, num_targets)
    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
class Tomics_and_Cell_Line_Model(nn.Module):
    def __init__(self, simple_model, cell_line_model, num_targets = None):
        super(Tomics_and_Cell_Line_Model, self).__init__()
        self.SimpleNN = simple_model
        self.cell_line = cell_line_model
        self.combo = nn.Linear(20, num_targets)
    def forward(self, x1, x2):
        x1 = self.SimpleNN(x1)
        x1 = F.relu(x1)
        x2 = self.cell_line(x2)
        x2 = F.relu(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.combo(x)
        return x
    

class Chemical_Structure_Model(nn.Module):
    """
    Simple 3-Layer FeedForward Neural Network
    
    For more info: https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold\
    /blob/main/final/Best%20LB/Training/3-stagenn-train.ipynb
    """
    def __init__(self, num_features = None, num_targets = None):
        super(Chemical_Structure_Model, self).__init__()
        #layer 1
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 120))
        self.batch_norm1 = nn.BatchNorm1d(120)
        self.dropout1 = nn.Dropout(0.3910762221579472)
        #layer 2
        self.dense2 = nn.utils.weight_norm(nn.Linear(120, 103))
        self.batch_norm2 = nn.BatchNorm1d(103)
        self.dropout2 = nn.Dropout(0.3284304805669022)

        # output layer
        self.output = nn.Linear(103, num_targets)
        #self.dense4 = nn.utils.weight_norm(nn.Linear(48, num_targets))
        
    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        
        x = F.leaky_relu(self.dense2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)


        x = self.output(x)
    
        
        return x

class Modified_GE_Model(nn.Module):
    def __init__(self, original_model):
        super(Modified_GE_Model, self).__init__()
        self.SimpleNN = original_model.SimpleNN
        self.cell_line = original_model.cell_line
        self.combo = nn.Linear(in_features = 20, out_features= 20)
    
    def forward(self, x1in, x2in):
        x1 = self.SimpleNN(x1in)
        x2 = self.cell_line(x2in)
        x = torch.cat((x1, x2), dim = 1)
        x = self.combo(x)
        return x
