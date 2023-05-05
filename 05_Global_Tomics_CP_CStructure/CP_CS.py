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
import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')

from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    accessing_correct_fold_csv_files, 
    create_splits, 
    choose_device,
    checking_veracity_of_data, 
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
    pre_processing,
    set_parameter_requires_grad,
    smiles_to_array
)
from Helper_Models import (image_network, MyRotationTransform, modelCS)
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

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import re



import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
incl_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_CS'
#---------------------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")


file_name = 'erik10_all'
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
variance_thresh = 0
normalize_c = 'False'

npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c)

df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = npy_exists,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = save_npy,
        data_subset = file_name)
checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)


# ----------------------------------------- Prepping Chemical Structure Data ---------------------------------------#
# download dictionary which associates moa with a tensor

dict_moa = dict_splitting_into_tensor(training_set)
assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique()), "Training, validation and test sets have different labels"
assert df_train_features.shape[0] == df_train_features.dropna().shape[0], "NaNs in training set"


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_set)
validation_df, validation_labels = splitting(validation_set)
test_df, test_labels = splitting(test_set)



# make sure that the number of labels is equal to the number of inputs
inputs_equalto_labels_check(training_df, train_labels, validation_df, validation_labels, test_df, test_labels)



num_classes = len(np.unique(df_train_labels["moa"]))

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



# load individual models
print("Loading Pretrained Models...")
modelCS = modelCS
modelCS.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'Chemical_Structure')['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'Cell_Painting' + '.pt')['model_state_dict'])

# -----------------------------------------Prepping Ensemble Model ---------------------#
class CS_CP_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, paths_df,  labels_CID, dict_moa, transform = None, im_norm = None):
        self.compound_df = compound_df
        self.paths_df = paths_df
        self.labels_CID = labels_CID
        self.dict_moa = dict_moa
        self.transform = transform
        self.im_norm = im_norm
    
    def __len__(self):
        return len(self.labels_CID)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        CID, label  = self.labels_CID.iloc[idx] 
        image = channel_5_numpy(self.paths_df, idx, self.im_norm)
        smile_string = self.compound_df["SMILES"][self.compound_df["Compound_ID"]== CID] 
        if smile_string.shape[0] > 1:
            row_num = smile_string.shape[0]
            selection = int(np.random.randint(0,row_num) - 1)
            smile_string = smile_string.iloc[selection]
            #print("We have an enantiomer")
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
        return image.float(), compound_array.float(), label_tensor.float() # returns 
        
# ---------------------------------------- Prepping Cell Painting Dataset ---------------------#
rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation for CP Images
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])

# extract compound IDs
test_data_lst= list(test_set["Compound_ID"].unique())
train_data_lst= list(training_set["Compound_ID"].unique())
valid_data_lst= list(validation_set["Compound_ID"].unique())

cmpd_id_overlap_check(train_data_lst, valid_data_lst, test_data_lst)

# downloading paths to all of the images
paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

# 
training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)



# importing data normalization pandas dataframe
pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')

# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CSCP = CS_CP_Dataset(training_set, paths_v1v2, df_train_labels, dict_moa,transform = train_transforms, im_norm= pd_image_norm)
valid_dataset_CSCP = CS_CP_Dataset(validation_set, paths_v1v2, df_val_labels, dict_moa, transform = train_transforms, im_norm= pd_image_norm)
test_dataset_CSCP = CS_CP_Dataset(test_set, paths_v1v2, df_test_labels, dict_moa, transform = train_transforms, im_norm= pd_image_norm)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSCP, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSCP, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSCP, **params)

# create a model combining both models
class CS_CP(nn.Module):
    def __init__(self, modelCP, modelCS):
        super(CS_CP, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.linear_layer1 = nn.Linear(int(1280 + 64), 128)
        self.selu = nn.SELU()
        self.Dropout = nn.Dropout(p = 0.3)
        self.linear_layer2 = nn.Linear(128,10)
       
    def forward(self, x1, x2):
        x1 = self.modelCP(x1)
        x2 = self.modelCS(x2)
        x  = torch.cat((torch.squeeze(x1), x2), dim = 1)
        x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer2(x)
        return output

model = CS_CP(modelCP, modelCS)

# optimizer_algorithm
#cnn_optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
yn_class_weights = True
class_weights = apply_class_weights_CL(training_set, dict_moa, device)
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, class_weights=class_weights)

# --------------------------------- Training, Test, Validation, Loops --------------------------------#



set_parameter_requires_grad(model, feature_extracting = True)

learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = True
class_weights = apply_class_weights_CL(training_set, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                  class_weights=class_weights)
my_lr_scheduler = 'false'
num_epochs_fs = 10
#n_epochs, optimizer, model, loss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false")
rain_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
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
              val_str = 'fe',
              early_patience = 50)

print('Fine Tuning in Progress')
set_parameter_requires_grad(model, feature_extracting = False)
learning_rate = 0.5e-6
my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, 
                        milestones=[8, 16, 22, 28, 32, 36], # List of epoch indices
                        gamma = 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
# loss_function
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                 class_weights=class_weights)

#----------------------------------------------------- Training and validation ----------------------------------#

train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
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
              early_patience = 10)
#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CS_CP
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model, 
                    test_loader = test_generator, 
                    device = device)
# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images')


#-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = normalize_c,
                    yn_class_weights = incl_class_weights,
                    learning_rate = learning_rate, 
                    elapsed_time = elapsed_time, 
                    num_epochs = num_epochs,
                    loss_fn = loss_fn,
                    all_labels = all_labels,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    variance_thresh = variance_thresh,
                    pixel_size = 256,
                    loss_fn_train = loss_fn_train)
