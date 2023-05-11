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
    LogScaler,
    pre_processing,
    accessing_all_folds_csv
)

start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = "DeepInsight"

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")


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


# not deep   50x50
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p= 0.23753380635366567)
        self.fc1 = nn.Linear(in_features=64*25*25, out_features=323)
        self.fc2 = nn.Linear(in_features=323, out_features= 654)
        self.fc3 = nn.Linear(in_features=654, out_features=10)

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
# ------------------------------------------------------------------------------------------------------------------------------
'''
file = input("Which file would you like to use? (Options: tian10, erik10, erik10_hq, erik10, erik10_hq_dos, erik10_dos, cyc_adr, cyc_dop):")
variance_threshold = input("What variance threshold would you like to use? (Options: 0 - 1.2):")
normalize = input("Would you like to normalize the data? (Options: True, False):")
'''  

file_name = "erik10_hq_8_12"
for fold_int in range(0,1):
    print(f'Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)
    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    variance_thresh = 0
    normalize_c = 'False'
    npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c, five_fold = 'True')
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


    pixel_size = (50,50)
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

    net = torchvision.models.squeezenet1_1(pretrained=True, progress=True)
    num_classes = len(df_train_labels)
    net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), 
                                stride=(1,1))



    max_epochs = 1
            
    
    model = DeepInsight_Model()


    batch_size = 50

    trainset = DI_Dataset(X_train_tensor, y_train_tensor, dict_moa)
    train_generator = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    validset = DI_Dataset(X_val_tensor, y_val_tensor, dict_moa)
    valid_generator = DataLoader(validset, batch_size=batch_size, shuffle=False)

    testset = DI_Dataset(X_test_tensor, y_test_tensor, dict_moa)
    test_generator = DataLoader(testset, batch_size=batch_size, shuffle=False)



    # In[50]:

    # If applying class weights
    yn_class_weights = True
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_CL(df_train_labels_moa, dict_moa, device)
    learning_rate = 0.0005706036846126339
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,  step_size=19, gamma= 0.8432222781730062)
    yn_class_weights = 'False'
    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'false'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                    loss_fn_train_str = loss_fn_train_str, 
                                                    class_weights=class_weights,
                                                    smoothing = 0.24296117253545618,
                                                    alpha = 0.20730849127124368 )

    #------------------------------   Calling functions --------------------------- #
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=train_generator, 
                valid_loader=valid_generator,
                my_lr_scheduler = my_lr_scheduler,
                model_name=model_name,
                device = device,
                val_str = 'f1',
                early_patience = 100)
    #--------------------------------- Assessing model on test data ------------------------------#
    model_test = DeepInsight_Model()
    model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')['model_state_dict'])
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model_test,
                                            test_loader = test_generator,
                                            device = device)

    #---------------------------------------- Visual Assessment ---------------------------------# 
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

    # results_assessment(all_predictions, all_labels, moa_dict)

    #-------------------------------- Writing interesting info into terminal ------------------------# 

    end = time.time()

    elapsed_time = program_elapsed_time(start, end)

    create_terminal_table(elapsed_time, all_labels, all_predictions)
    upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = "False",
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
                        loss_fn_train = loss_fn_train)
