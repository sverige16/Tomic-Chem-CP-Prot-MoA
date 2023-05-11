from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem


from pyDeepInsight import ImageTransformer, Norm2Scaler
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

from Erik_alll_helper_functions import checking_veracity_of_data, dict_splitting_into_tensor, val_vs_train_loss, val_vs_train_accuracy, EarlyStopper
from Erik_alll_helper_functions import conf_matrix_and_class_report, program_elapsed_time, dict_splitting_into_tensor
from Erik_alll_helper_functions import apply_class_weights, set_parameter_requires_grad, LogScaler, smiles_to_array, create_splits
from Erik_alll_helper_functions import pre_processing, accessing_correct_fold_csv_files, channel_5_numpy_CID, splitting

from Helper_Models import DeepInsight_Model, Chem_Dataset, Reducer_profiles, image_network, MyRotationTransform
from efficientnet_pytorch import EfficientNet 



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

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
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
    CP_driving_code
)
from Helper_Models import (SimpleNN_Model, 
                           Transcriptomic_Profiles_Cell_Lines, 
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           Chemical_Structure_Model)

from Helper_Models import (image_network, MyRotationTransform, Chemical_Structure_Model)
from efficientnet_pytorch import EfficientNet 


# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
incl_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

class CS_CP_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, paths_df, tprofiles_df, labels_CID, dict_moa, transform = None, im_norm = None):
        self.compound_df = compound_df
        self.paths_df = paths_df
        self.tprofiles_df = tprofiles_df
        self.labels_CID = labels_CID
        self.dict_moa = dict_moa
        self.transform = transform
        self.im_norm = im_norm
    
    def __len__(self):
        return len(self.labels_CID)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        tprofile = self.tprofiles_df[idx]
        CID, label  = self.labels_CID.iloc[idx] 
        image = channel_5_numpy_CID(self.paths_df, CID, self.im_norm)
        smile_string = self.compound_df["SMILES"][self.compound_df["Compound_ID"]== CID] 
        if smile_string.shape[0] > 1:
            smile_string = smile_string.iloc[0]
            print("We have an enantiomer")
        if type(smile_string) == pd.Series:
            smile_string = smile_string.values[0]
        elif type(smile_string) == str:
            smile_string = smile_string
        else:
            print("We have a problem")
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        return tprofile, compound_array.float(), image.float(), label_tensor.float() # returns 

     
# create a model combining both models
class CS_CP_GE_Model(nn.Module):
    def __init__(self, modelCP, modelDI, modelCS):
        super(CS_CP_GE_Model, self).__init__()
        self.modelCP = modelCP
        self.modelDI = modelDI
        self.modelCS = modelCS
        self.linear_layer1 = nn.Linear(int(10 + 10 + 10), 25)
        self.selu = nn.SELU()
        self.Dropout = nn.Dropout(p = 0.5)
        self.linear_layer2 = nn.Linear(25,10)
       
    def forward(self, x1, x2, x3):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        x1 = self.modelCP(x1)
        x2 = self.modelDI(x2)
        x3 = self.modelCS(x3)
        x  = torch.cat((x1, x2, x3), dim = 1)
        x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer2(x)
        return output

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
#---------------------------------------------------------------------------------------------------------------------------------------#

# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 12
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# -----------------------------------------Prepping Ensemble Model ---------------------#


file_name = "erik10_hq_8_12"
fold_num = 0
train_np_GE, valid_np_GE, test_np_GE, num_classes, dict_cell_lines, num_cell_lines, dict_moa = GE_driving_code(file_name, fold_num)
training_df_CP, validation_df_CP, test_df_CP, paths_v1v2_CP, df_train_labels_CP, dict_moa, pd_image_norm_CP, df_val_labels_CP, df_test_labels_CP, num_classes, cmpd_training_set, cmpd_validation_set, cmpd_test_set = CP_driving_code(file_name, fold_num)

# download csvs with all the data pre split

# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 10
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# load individual models
print("Loading Pretrained Models...")

# ---------------------------------------- Prepping Cell Painting Dataset ---------------------#
rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation for CP Images
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])

# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CS_CP_GE = CS_CP_GE_Dataset(paths_v1v2_CP, train_np_GE, training_df_CP, df_train_labels_CP, dict_moa, transform = train_transforms, im_norm= pd_image_norm_CP)
valid_dataset_CS_CP_GE = CS_CP_GE_Dataset(paths_v1v2_CP, valid_np_GE, validation_df_CP, df_val_labels_CP, dict_moa, transform = train_transforms, im_norm= pd_image_norm_CP)
test_dataset_CS_CP_GE = CS_CP_GE_Dataset(paths_v1v2_CP, test_np_GE, test_df_CP, df_test_labels_CP, dict_moa, transform = train_transforms, im_norm= pd_image_norm_CP)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CS_CP_GE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CS_CP_GE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CS_CP_GE, **params)

# create a model combining both models
cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])

model_CP_GE = CS_CP_GE_Model(modelCP, modelGE)
   
# --------------------------------- Training, Test, Validation, Loops --------------------------------#

