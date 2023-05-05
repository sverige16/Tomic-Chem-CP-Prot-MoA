import torch.nn as nn
import pickle 
import numpy as np


#!/usr/bin/env python
# coding: utf-8

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
# Neptune
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve,log_loss, accuracy_score, f1_score, classification_report
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.ensemble import VotingClassifier
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
import time
import joblib

from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import precision_recall_curve,log_loss
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import os
import pandas as pd
import numpy as np
import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
from torchsummary import summary
nn._estimator_type = "classifier"
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
    channel_5_numpy,
    splitting,
    cmpd_id_overlap_check, 
    inputs_equalto_labels_check,
)

start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = "IGTD"
print("Begin Training")

# ------------ Load .pkl file ------------ #
# Euclidean distances with absolute error
'''
with open('/scratch2-shared/erikep/Results/Euc_full/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "euclid"

with open('/scratch2-shared/erikep/Results/Euc_full/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/Pear/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"

with open('/scratch2-shared/erikep/Results/Pear/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/hq_Pear/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"

with open('/scratch2-shared/erikep/Results/hq_Pear/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/erik10_hq_8_12_Pearson/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"
    file_name = "erik_hq_8_12"
    variance_thresh = 0
    norm = False

with open('/scratch2-shared/erikep/Results//erik10_hq_8_12_Pearson/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
generated_images = np.transpose(data, (2, 0, 1))

# ------------- Load Labels -------------- #
with open('/scratch2-shared/erikep/Results/labels_hq_moadict.pkl', 'rb') as f:
    all_labels = pickle.load(f)
train_labels, valid_labels, test_labels, dict_moa, dict_indexes = all_labels

with open('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/IGTD_erik10_hq_8_12_splits.pkl', 'rb') as f:
    index_splits = pickle.load(f)
labels = pd.concat([train_labels, valid_labels, test_labels], axis=0).reset_index(drop=True)
'''
last_training_index = train_labels.shape[0]
last_validat_index = last_training_index + valid_labels.shape[0]
last_test_index = generated_images.shape[0]
# split generated images into train, test and validation
train_images = generated_images[:last_training_index]
valid_images = generated_images[last_training_index: last_validat_index]
test_images = generated_images[last_validat_index: last_test_index]

'''
for fold_int in range(2,5):
    print(f'Fold Iteration: {fold_int}')
    index_split_n =index_splits[fold_int]
    # split generated images into train, test and validation
    train_images = generated_images[index_split_n["train"]]
    valid_images = generated_images[index_split_n["valid"]]
    test_images = generated_images[index_split_n["test"]]

    # split generated labels into train, test and validation
    train_labels = labels.iloc[index_split_n["train"]].reset_index(drop=True)
    valid_labels = labels.iloc[index_split_n["valid"]].reset_index(drop=True)
    test_labels = labels.iloc[index_split_n["test"]].reset_index(drop=True)
    '''train_images = generated_images[:train_labels.shape[0]]
    valid_images = pd.DataFrame(generated_images[(int(train_labels.shape[0])) + 1  : (int(train_labels.shape[0]) + 1  + int(valid_labels.shape[0]))]).reset_index(drop=True)
    test_images = pd.DataFrame(generated_images[(int(train_labels.shape[0]) + int(valid_labels.shape[0])+ 1):train_labels.shape[0] + valid_labels.shape[0] + test_labels.shape[0]]).reset_index(drop=True)
    '''

    # checking that the number of images and labels are the same
    inputs_equalto_labels_check(train_images, 
                                train_labels, 
                                valid_images, 
                                valid_labels, 
                                test_images, 
                                test_labels)
    #assert samples[last_test_index -1 ] ==  '2706'

    using_cuda = True
    device = choose_device(using_cuda)


    class IGTD_profiles(torch.utils.data.Dataset):
        def __init__(self, feat_img, labels, dict_moa, transform = None):
            self.X = feat_img
            self.y = labels
            self.dict_moa = dict_moa
            self.transform = transform

        def __getitem__(self, index):
            label = dict_moa[self.y[index]]
            img = torch.tensor([self.X[index]], dtype=torch.float)
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float)

        def __len__(self):
            return len(self.X)
    class IGTD_Model(nn.Module):
        def __init__(self):
            super(IGTD_Model, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels= 12, kernel_size=(2,2), padding=1)
            self.conv2 = nn.Conv2d(in_channels= 12, out_channels=16, kernel_size=(2,2), padding=1)
            self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
            self.dropout = nn.Dropout(p=0.2000578900281615)
            self.fc1 = nn.Linear(16*4* 82, 2447)
            self.fc2 = nn.Linear(2447,  230)
            self.fc3 = nn.Linear( 230, 10)
            
        def forward(self, x):
            #x = x.unsqueeze(1)
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            #x = self.conv2_bn(x)
            x = nn.functional.relu(x)
            x = self.pool(x)
            x = self.dropout(x)
            x = x.view(-1, 16*4* 82
                    )
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            return x     
    #DI_model = DeepInsight_Model(net)     
    model = IGTD_Model()


    batch_size = 50
    train_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma = (0.1, 0.2))

    trainset = IGTD_profiles(train_images, train_labels["moa"], dict_moa)
    training_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    validset = IGTD_profiles(valid_images, valid_labels["moa"], dict_moa)
    validation_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    testset = IGTD_profiles(test_images, test_labels["moa"], dict_moa)
    test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)



    # In[50]:

    # If applying class weights

    max_epochs = 300
    yn_class_weights = False
    learning_rate = 0.00046372531356603854
    optimizer = torch.optim.Adam(model.parameters(),  lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,  step_size=18, gamma=0.7910577406713815)
    yn_class_weights = 'False'
    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'false'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                    loss_fn_train_str = loss_fn_train_str, 
                                                    )




    # In[52]:
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
                early_patience = 100)

    #--------------------------------- Assessing model on test data ------------------------------#
    model_test = IGTD_Model()
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