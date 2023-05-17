from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem


#from pyDeepInsight import ImageTransformer, Norm2Scaler
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
import warnings
warnings.filterwarnings('ignore')



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
import warnings


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
    set_parameter_requires_grad,
    returning_smile_string,
    dubbelcheck_dataset_length
)

from Helper_Models import (SimpleNN_Model, 
                           Transcriptomic_Profiles_Cell_Lines, 
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           image_network,
                           Chemical_Structure_Model,
                           Modified_GE_Model)
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter('error')

class CS_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, tprofiles_df, split_sets, dict_moa, dict_cell_line, checking_mechanism):
        self.compound_df = compound_df
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_moa = dict_moa
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
    def __len__(self):
        check_criteria = self.check
        assert len(self.tprofiles_df) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.tprofiles_df)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        t_profile = self.tprofiles_df.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.tprofiles_df.iloc[idx, -1]
        CID = self.split_sets["Compound ID"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        cell_line_key = self.split_sets["cell_iname"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        CID = CID.iloc[0]
        cell_line_key = cell_line_key.iloc[0]
        t_cell_line = torch.tensor(self.dict_cell_line[cell_line_key])
        smile_string = returning_smile_string(self.compound_df,CID)
        compound_array = smiles_to_array(smile_string)
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        label_tensor = torch.from_numpy(self.dict_moa[moa_key])                  # convert label to number
        t_profile_features = torch.tensor(t_profile) 
        return compound_array.float(), t_profile_features, t_cell_line.float(), label_tensor.float() # returns 
        
class CS_GE_Model(nn.Module):
    def __init__(self, modelCS, modelGE):
        super(CS_GE_Model, self).__init__()
        self.modelCS = torch.nn.Sequential(*list(modelCS.children())[:-1])
        self.modelGE = modelGE
        self.linear_layer_pre = nn.Linear(int(103 + 20), 134)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.batch_normpre = nn.BatchNorm1d(134)
        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=134, out_features=10, bias=True))

    #s: {'init_nodes': 134, 'n_layers': 1, 'optimizer': 'RMSprop', 'lr': 0.07636505639465482, 'scheduler': 'StepLR', 'step_size': 6, 'gamma': 0.735004577042182, 'yn_class_weights': False, 'loss_fn': 'cross', 'loss_train_fn': 'false'}. Best is trial 73 with value: 0.495196351934163.
    def forward(self, x1in, x2in, x3in):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        if x1in.shape[1] != 2048:
            raise ValueError("The input shape for the compound is not correct")
        x1 = self.modelCS(x1in)
        x2 = self.modelGE(x2in, x3in)
        x  = torch.cat((x1, x2), dim = 1)
        x = self.batch_normpre(self.leaky_relu(self.linear_layer_pre(x)))
        output = self.additional_layers(x)
        return output


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
model_name = 'CS_GE'
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
#clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
#clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")



batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 0.0000195589 
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2,
         'shuffle' : True}
          
using_cuda = True 
device = choose_device(using_cuda)         


file_name = "erik10_hq_8_12"
fold_num = 0
train_np, valid_np, test_np, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training, L1000_validation, L1000_test = GE_driving_code(file_name, fold_num)



# load individual models
print("Loading Pretrained Models...")
cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)


 

modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modelCS = Chemical_Structure_Model(num_features = 2048, num_targets = 10)
modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = 0), map_location=torch.device('cpu'))['model_state_dict'])
modified_GE_model = Modified_GE_Model(modelGE)
model = CS_GE_Model(modelCS, modified_GE_model)
    

# -----------------------------------------Prepping Ensemble Model ---------------------#

# Create datasets with relevant data and labels
training_dataset_CSGE = CS_GE_Dataset(training_set_cmpds, train_np, L1000_training, dict_moa, dict_cell_lines, ["train" , "GE", fold_num])
valid_dataset_CSGE = CS_GE_Dataset(validation_set_cmpds, valid_np, L1000_validation, dict_moa, dict_cell_lines, ["valid" , "GE", fold_num])
test_dataset_CSGE = CS_GE_Dataset(test_set_cmpds, test_np, L1000_test, dict_moa, dict_cell_lines, ["test" , "GE", fold_num])

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSGE, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSGE, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSGE, **params)



set_parameter_requires_grad(model, feature_extracting = True, added_layers = 0)

learning_rate = 0.07636505639465482
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = True
class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                  class_weights=class_weights)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.735004577042182)
num_epochs_fe = 10
#n_epochs, optimizer, model, lo ss_fn, loss_fn_str, train_loader, valid_loader, my_lr_scheduler, device, model_name, loss_fn_train = "false")
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = num_epochs_fe,
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
              early_patience = 0)


set_parameter_requires_grad(model, feature_extracting = False)
learning_rate = 0.07636505639465482/10
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
# loss_function
yn_class_weights = True
class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
# choosing loss_function 
loss_fn_str = 'cross'
loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                  class_weights=class_weights)
#my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.735004577042182)
my_lr_scheduler = 'false'
num_epochs_ft = 10
#----------------------------------------------------- Training and validation ----------------------------------#

train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = num_epochs_ft,
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
              early_patience = 0)

#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CS_GE_Model(modelCS, modified_GE_model)
model_test.load_state_dict(torch.load('/home/jovyan/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model, 
                    test_loader = test_generator, 
                    device = device)
# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/saved_images')


#-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = True,
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
                    variance_thresh = 0,
                    pixel_size = 0,
                    loss_fn_train = loss_fn_train)