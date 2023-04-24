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
import re

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import apply_class_weights, accessing_correct_fold_csv_files, create_splits, extract_tprofile
from Erik_alll_helper_functions import checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss
from Erik_alll_helper_functions import val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import pre_processing, create_terminal_table, upload_to_neptune, dict_splitting_into_tensor
from Erik_alll_helper_functions import tprofiles_gc_too_func, set_bool_hqdose

# In[63]:


using_cuda = True
hidden_size = 1000


# In[39]:

model_name = 'SimpleNN'

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

import torch
'''
def differentiable_f1_loss(y_true, y_pred, class_weights=None):
    smooth = 1e-7

    # Convert predictions to binary by rounding
    y_pred = torch.round(torch.clamp(y_pred, 0, 1))

    # Calculate true positives, false positives, and false negatives
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    # Calculate continuous precision and recall
    p = tp / (tp + fp + smooth)
    r = tp / (tp + fn + smooth)

    # Calculate the differentiable F1 score
    f1 = 2 * p * r / (p + r + smooth)
    f1[torch.isnan(f1)] = 0

    # Apply class weights if provided
    if class_weights is not None:
        f1 = f1 * class_weights
        f1 /= torch.sum(class_weights)

    # Calculate the loss as 1 - F1 score
    return 1 - torch.mean(f1)
'''

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=-1)
        #targets_one_hot = torch.zeros_like(probs)
        #targets_one_hot.scatter_(1, targets.view(-1, 1), 1)
        targets_one_hot = targets
        
        pt = torch.where(targets_one_hot == 1, probs, 1 - probs)
        ce_loss = -torch.log(pt)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(torch.sum(focal_loss, dim=-1))
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
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
learning_rate = 0.0005226706526289529
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

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
# Creating a  dictionary of the one hot encoded labels
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



# In[49]:


# generator: training
# create a subset with only train indices

# create generator that randomly takes indices from the training set
training_dataset = Transcriptomic_Profiles(profiles_gc_too_train, L1000_training, dict_moa)
training_generator = torch.utils.data.DataLoader(training_dataset, **params)

validation_dataset = Transcriptomic_Profiles(profiles_gc_too_valid, L1000_validation, dict_moa)
validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)

test_dataset = Transcriptomic_Profiles(profiles_gc_too_test, L1000_test, dict_moa)
test_generator = torch.utils.data.DataLoader(Transcriptomic_Profiles(profiles_gc_too_test, L1000_test, dict_moa), **params)


class SimpleNN_Model(nn.Module):
    """
    Simple 3-Layer FeedForward Neural Network
    
    For more info: https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold\
    /blob/main/final/Best%20LB/Training/3-stagenn-train.ipynb
    """
    def __init__(self, num_features = None, num_targets = None, hidden_size = None):
        super(SimpleNN_Model, self).__init__()
        #self.batch_norm1 = nn.BatchNorm1d(num_features)
        #self.dropout1 = nn.Dropout(0.29323431985537163)
        #layer 1
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 64))
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.40473107898476735)
        #layer 2
        self.dense2 = nn.utils.weight_norm(nn.Linear(64, 43))
        self.batch_norm2 = nn.BatchNorm1d(43)
        self.dropout2 = nn.Dropout(0.30761037332988056)
        #layer 3
        self.dense3 = nn.Linear(43, 103)
        self.batch_norm3 = nn.BatchNorm1d(103)
        self.dropout3 = nn.Dropout( 0.31140834347665325)

        # output layer
        self.output = nn.Linear(103,10)
        #self.dense4 = nn.utils.weight_norm(nn.Linear(48, num_targets))

    def forward(self, x):
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

        
        return x


# In[66]:

yn_class_weights = True
class_weights = apply_class_weights(training_set, device)
model = SimpleNN_Model(num_features = 978, num_targets= num_classes, hidden_size= hidden_size)
optimizer = torch.optim.RMSprop(model.parameters(),  weight_decay=WEIGHT_DECAY, lr=learning_rate)
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,  step_size = 28, gamma=0.25568940073373997)
from ols import OnlineLabelSmoothing
#loss_fn = torch.nn.BCEWithLogitsLoss(weight = class_weights)
loss_fn_train = "false"
#loss_fn_train = OnlineLabelSmoothing(alpha =  0.7108530368235001, n_classes=num_classes, smoothing = 0.2232899344804938).to(device=device)
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
#loss_fn = torch.nn.CrossEntropyLoss(weight = class_weights)

# In[67]:


# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
max_epochs = 100 # number of epochs we are going to run 
# apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

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
    early_stopper = EarlyStopper(patience=5, min_delta=0.0001)
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
            loss = loss_fn(outputs, labels)
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
        val_correct, val_total, val_loss, best_val_loss_upd, val_f1_score = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, f1 score: {val_f1_score} ')
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
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss = loss_fn(outputs, labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        pred_cpu = torch.cat(predictions).cpu()
        labels_cpu =  torch.cat(all_labels).cpu()
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
    return correct, total, avg_val_loss, best_val_loss,  f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro')



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
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss = loss_fn(outputs, labels)
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


train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              loss_fn_train= loss_fn_train,
              my_lr_scheduler= my_lr_scheduler)



val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images')

#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = SimpleNN_Model(num_features = 978, num_targets= num_classes, hidden_size= hidden_size)  
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = model_test,
                                          loss_fn = loss_fn, 
                                          test_loader = test_generator)



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