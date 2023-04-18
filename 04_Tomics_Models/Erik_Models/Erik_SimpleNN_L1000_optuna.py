#!/usr/bin/env python
# coding: utf-8

# In[36]:


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

import torch.nn.functional as F
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,log_loss, f1_score, accuracy_score
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN
from sklearn.feature_selection import VarianceThreshold

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import time
import heapq
import optuna

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import apply_class_weights, accessing_correct_fold_csv_files, create_splits
from Erik_alll_helper_functions import dict_splitting_into_tensor, extract_tprofile, EarlyStopper, val_vs_train_loss
from Erik_alll_helper_functions import val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import tprofiles_gc_too_func, create_terminal_table, upload_to_neptune
from Erik_alll_helper_functions import set_bool_hqdose, set_bool_npy, find_two_lowest_numbers



using_cuda = True
hidden_size = 1000
model_name = 'SimpleNN_L1000_optuna'


# In[39]:


# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")



class Transcriptomic_Profiles(torch.utils.data.Dataset):
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


batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 5e-3
num_feat = 0
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2} 
          
if using_cuda:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
else:
    device = torch.device('cpu')
print(f'Training on device {device}. ' )



file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

dict_moa = dict_splitting_into_tensor(training_set)

# checking that no overlap in sig_id exists between training, test, validation sets
inter1 = set(list(L1000_training["sig_id"])) & set(list(L1000_validation["sig_id"]))
inter2 = set(list(L1000_training["sig_id"])) & set(list(L1000_test["sig_id"]))
inter3 = set(list(L1000_validation["sig_id"])) & set(list(L1000_test["sig_id"]))
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There is an intersection between the training, validation and test sets")

# shuffling training and validation data
L1000_training = L1000_training.sample(frac = 1, random_state = 1)
L1000_validation = L1000_validation.sample(frac = 1, random_state = 1)
L1000_test = L1000_test.sample(frac = 1, random_state = 1)

print("extracting training transcriptomes")
profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
print("extracting validation transcriptomes")
profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
print("extracting test transcriptomes")
profiles_gc_too_test = tprofiles_gc_too_func(L1000_test, clue_gene)

num_classes = len(L1000_training["moa"].unique())

# create generator that randomly takes indices from the training set
training_dataset = Transcriptomic_Profiles(profiles_gc_too_train, L1000_training, dict_moa)
training_generator = torch.utils.data.DataLoader(training_dataset, **params)

# create generator that randomly takes indices from the validation set
validation_dataset = Transcriptomic_Profiles(profiles_gc_too_valid, L1000_validation, dict_moa)
validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

test_dataset = Transcriptomic_Profiles(profiles_gc_too_test, L1000_test, dict_moa)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)



# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
max_epochs = 30 # number of epochs we are going to run 
# apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

# --------------------------Function to perform training, validation, testing, and assessment ------------------


# --------------------------Function to perform training, validation, testing, and assessment ------------------

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, my_lr_scheduler, loss_fn_train = "false"):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    early_stopper = EarlyStopper(patience=15, min_delta=0.0001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    if loss_fn_train != "false":
        loss_fn_train.train()
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for tprofiles, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(tprofiles)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
       
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            loss.backward()
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, Accuracy: {val_accuracy} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                                

def validation_loop(model, loss_fn, valid_loader, best_val_loss):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    model = model.to(device)
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for tprofiles, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(tprofiles)
            #probs = torch.nn.Softmax(outputs)
            #loss = loss_fn(outputs,labels)
            loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            m = torch.nn.Softmax(dim=1)
            pred_cpu = torch.cat(predictions).cpu()
            labels_cpu =  torch.cat(all_labels).cpu()
            torch.save(
                {   'predict_proba' : m(torch.cat(predict_proba)),
                    'predictions' : pred_cpu.numpy(),
                    'labels_val' : labels_cpu.numpy(),
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val,
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss


def test_loop(model, loss_fn, test_loader):
    '''
    Assessing trained model on test dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    test_loader: generator creating batches of test data
    '''
    model = model.to(device)
    model.eval()
    loss_test = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on test data.
        for compounds, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = True):
            # Move to device MAY NOT BE NECESSARY
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)

            # Assessing outputs
            outputs = model(compounds)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
            #loss = loss_fn(outputs,labels)
            loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss_test += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == torch.max(labels, 1)[1]).sum())
            #print(f' Predicted: {predicted.tolist()}')
            #print(f' Labels: {predicted.tolist()}')
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + torch.max(labels, 1)[1].tolist()
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss, all_predictions, all_labels


#---------------------------------------- Visual Assessment ---------------------------------# 



def define_model(trial, num_feat, num_classes):
    # optimizing hidden layers, hidden units and drop out ratio
    n_layers = trial.suggest_int('n_layers', 1, 4)
    layers = []
    in_features = num_feat
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(out_features))
        p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(out_features, num_classes))
    return nn.Sequential(*layers)

def objectiv(trial, num_feat, num_classes, training_generator, validation_generator):
    # generate the model
    model = define_model(trial, num_feat, num_classes).to(device)
    
    # generate the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_uniform("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))

    class_weights = apply_class_weights(training_set, device)
    loss_fn = torch.nn.CrossEntropyLoss(class_weights)

    from ols import OnlineLabelSmoothing
    loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3)).to(device=device)

    max_epochs = 1000

    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = scheduler)
    
  

    lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return (lowest1 + lowest2)/2

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = num_classes, 
                                      training_generator= training_generator, 
                                      validation_generator = validation_generator), 
                                      n_trials=100)

#-------------------------------- Writing interesting info into terminal ------------------------# 

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