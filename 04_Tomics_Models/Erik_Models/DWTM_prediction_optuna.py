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
import optuna
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
import numpy as np
import cv2
import os
start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")


file_name = "erik10_hq_8_12"
def load_images(folder_path, train_range, test_range, validation_range):
    def read_image(file_path):
        img = cv2.imread(file_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def read_images_in_range(r):
        images = []
        for i in r:
            file_name = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(file_name):
                images.append(read_image(file_name))
            else:
                print(f"File {file_name} not found.")
        return np.array(images)

    train_set = read_images_in_range(train_range)
    test_set = read_images_in_range(test_range)
    validation_set = read_images_in_range(validation_range)

    return train_set, test_set, validation_set

with open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/" + file_name + "labels_moa_dict" +'_' + 'fold0'+ ".pkl", 'rb') as f:
    all_labels = pickle.load(f)
train_labels, valid_labels, test_labels, dict_moa, dict_indexes = all_labels


# checking that the number of images and labels are the same
# Example usage:
folder_path = '/scratch2-shared/erikep/DWTM/ImageDataset/'
train_range = range(dict_indexes["train"][0], dict_indexes["train"][1]+1)  # Images 1-100 for training
validation_range = range(dict_indexes["valid"][0],dict_indexes["valid"][1]+1)  # Images 101-125 for testing
test_range = range(dict_indexes["test"][0], dict_indexes["test"][1]+1)  # Images 126-150 for validation

train_images, test_images, valid_images = load_images(folder_path, train_range, test_range, validation_range)
inputs_equalto_labels_check(train_images, 
                            train_labels, 
                            valid_images, 
                            valid_labels, 
                            test_images, 
                            test_labels)
using_cuda = True
device = choose_device(using_cuda)

class DWTM_profiles(torch.utils.data.Dataset):
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
        img = img.reshape(3, 128, 128)
        return img, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)

#DI_model = DeepInsight_Model(net)     

model_name = 'DWTM'



batch_size = 50

trainset = DWTM_profiles(train_images, train_labels["moa"], dict_moa)
training_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = DWTM_profiles(valid_images, valid_labels["moa"], dict_moa)
validation_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = DWTM_profiles(test_images, test_labels["moa"], dict_moa)
test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)



def objectiv(trial, num_feat, num_classes, training_generator, validation_generator, testing_generator):
    pretrained = trial.suggest_categorical('pretrained', [True, False])
    chosen_model = trial.suggest_categorical('model', ['resnet18', 'dense121'])
    if chosen_model == 'resnet18':
        model = models.resnet18(pretrained= pretrained)
        num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, len(dict_moa))
    else:
        model = models.densenet121(pretrained= pretrained)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# 
    
  
    model = model.to(device)             
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
        class_weights = apply_class_weights_CL(train_labels, dict_moa, device)
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
        

    max_epochs = 100

    
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
                early_patience = 12)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return max(val_f1_score_per_epoch)

storage = 'sqlite:///' + model_name + '.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = 10, 
                                      training_generator= training_generator, 
                                      validation_generator = validation_generator,
                                      testing_generator = test_generator), 
                                      n_trials=110)
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
