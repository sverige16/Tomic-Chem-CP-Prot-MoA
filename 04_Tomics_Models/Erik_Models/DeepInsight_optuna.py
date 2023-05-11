#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import numpy as np
import datetime
import time

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
# Neptune
import neptune.new as neptune


import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

import torchvision

#from pyDeepInsight import ImageTransformer, Norm2Scaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
import datetime
import time

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
# Neptune
import neptune.new as neptune
# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
import os
import pandas as pd
import numpy as np
import torch
import re
from DeepInsight_Image_Transformer import ImageTransformer
import optuna
import heapq

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
    adapt_training_loop, 
    adapt_validation_loop, 
    adapt_test_loop,
    checking_veracity_of_data,
    LogScaler,
    pre_processing
)
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = "DeepInsight"
print("Begin Training")

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# ------------------------------------------------------------------------------------------------------------------------------


file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)


checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
#variance_thresh = int(input("Variance threshold? (Options: 0 - 1.2): "))
#normalize_c = input("Normalize? (Options: True, False): ")
variance_thresh = 0
normalize_c = False
npy_exists = True
save_npy = False
df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = npy_exists,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = save_npy,
        data_subset = file_name)
checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)

df_train_labels_moa = df_train_labels
# save moa labels, remove compounds
df_train_labels = df_train_labels["moa"]
df_val_labels = df_val_labels["moa"]
df_test_labels = df_test_labels["moa"]

X_train = df_train_features.to_numpy().astype(np.float32)
X_val = df_val_features.to_numpy().astype(np.float32)
y_train = df_train_labels.values
y_val = df_val_labels.values
X_test = df_test_features.to_numpy().astype(np.float32)
y_test = df_test_labels.values


ln = LogScaler()
X_train_norm = ln.fit_transform(df_train_features.to_numpy().astype(float))
X_val_norm = ln.transform(df_val_features.to_numpy().astype(float))
X_test_norm = ln.transform(df_test_features.to_numpy().astype(float))


# In[51]:
#import umap 

distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    learning_rate='auto',
    perplexity=5,
    n_jobs=-1
)


#pixel_size = (10, 10)
#pixel_size = (20, 20)
#pixel_size = (30, 30)
pixel_size = (50,50)
#pixel_size = (100,100)
#pixel_size = (224,224)
it = ImageTransformer( 
    pixels=pixel_size)



# In[53]:

# fitting Image Transformer
print("Fitting Image Transformer...")
it.fit(X_train)
print("Transforming...")
X_train_img = it.transform(X_train_norm)
X_val_img = it.transform(X_val_norm)
X_test_img = it.transform(X_test_norm)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torchvision.models as models

preprocess = transforms.Compose([
    transforms.ToTensor()
])


X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
y_train_tensor = df_train_labels
X_val_tensor = torch.stack([preprocess(img) for img in X_val_img]).float()
y_val_tensor = df_val_labels

X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
y_test_tensor = df_test_labels

# not deep   50x50
class DeepInsight_Model(nn.Module):
    def __init__(self, dropout_rate, num_features1, num_features2):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p= dropout_rate)
        self.fc1 = nn.Linear(in_features=64*25*25, out_features=num_features1)
        self.fc2 = nn.Linear(in_features=num_features1, out_features= num_features2)
        self.fc3 = nn.Linear(in_features=num_features2, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# instantiate the model



class DI_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, dict_moa):
        self.X = X
        self.y = y
        self.dict_moa = dict_moa

    def __getitem__(self, index):
        label = self.dict_moa[self.y[index]]
        return self.X[index], torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)

batch_size = 50

trainset = DI_Dataset(X_train_tensor, y_train_tensor, dict_moa)
train_generator = DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = DI_Dataset(X_val_tensor, y_val_tensor, dict_moa)
valid_generator = DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = DI_Dataset(X_test_tensor, y_test_tensor, dict_moa)
test_generator = DataLoader(testset, batch_size=batch_size, shuffle=False)



# In[52]:
# --------------------------Function to perform training, validation, testing, and assessment ------------------


def objectiv(trial, num_feat, num_classes, training_generator, validation_generator, testing_generator):
    
    # generate the model
    num_features1 = trial.suggest_int('num_features1', 10, 1000)
    num_features2 = trial.suggest_int('num_features2', 10, 1000)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    model = DeepInsight_Model(num_features1=num_features1,
                              num_features2=num_features2,
                              dropout_rate=dropout_rate).to(device)
    
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
        class_weights = apply_class_weights_CL(df_train_labels_moa, dict_moa, device)
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'focal', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false','ols'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        

    max_epochs = 300

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
                early_patience = 20)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return max(val_f1_score_per_epoch)

storage = 'sqlite:///' + model_name + '.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = 10, 
                                      training_generator= train_generator, 
                                      validation_generator = valid_generator,
                                      testing_generator = test_generator), 
                                      n_trials=150)
print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/" + model_name + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()
