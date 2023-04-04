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

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import smiles_to_array,  accessing_correct_fold_csv_files, create_splits, dict_splitting_into_tensor, splitting
from Erik_alll_helper_functions import EarlyStopper,  val_vs_train_loss, val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import apply_class_weights

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
#apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#
# create Torch.dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, labels_df, dict_moa, transform=None):
        self.compound_labels = labels_df    # the entire length of the correct classes that we are trying to predict
        # print(self.img_labels)
        self.compound_df = compound_df        # list of indexes that are a part of training, validation, tes sets
        self.transform = transform       # any transformations done
        self.dict_moa = dict_moa

    def __len__(self):
        ''' The number of data points'''
        return len(self.compound_labels)      

    def __getitem__(self, idx):
        '''Retrieving the compound '''
        #print(idx)
        smile_string = self.compound_df["SMILES"][idx]      # returns smiles by using compound as keys
        #print(smile_string)
        compound_array = smiles_to_array(smile_string)
        #print(f' return from function: {compound}')
        #print(f' matrix: {compound_array}')
        label = self.compound_labels.iloc[idx]             # extract classification using index
        #print(f' label: {label}')
        #label = torch.tensor(label, dtype=torch.float)
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return compound_array.float(), label_tensor.long() # returns the image and the correct label

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")


# # Pseudo Code
# 1. download train, validation and test data
# 2. keep only columns with relevant data (compound name, smile strings, moa)
#     1. assert that dmso not in all data values
# 3. create function that gets produces a dictionary with all relevant moas and assigns a number to them
#     1. extract unique values
#     2. enumerate loop, add to growing dictionary, where name is the key.
# 4. set these values as moa
# 5. do one hot encoding
# 6. Fix torch_set so that it can handle pandas instead of an extra dictionary 

# donwload compound list for both v1 and v2
compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")


file_name = "erik10"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = 'False', 'False'
if re.search('hq', file_name):
    hq = 'True'
if re.search('8', file_name):
    dose = 'True'
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)


# download dictionary which associates moa with a tensor

dict_moa = dict_splitting_into_tensor(training_set)
assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique())

test_data_lst = list(test_set['Compound_ID'])
train_data_lst = list(training_set['Compound_ID'])
valid_data_lst = list(validation_set['Compound_ID'])

# check to make sure the compound IDs do not overlapp
inter1 = set(test_data_lst) & set(train_data_lst)
inter2 = set(test_data_lst) & set(valid_data_lst)
inter3 = set(train_data_lst) & set(valid_data_lst)
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_set)
validation_df, validation_labels = splitting(validation_set)
test_df, test_labels = splitting(test_set)



batch_size = 200
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 2} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )


# Create datasets with relevant data and labels
training_dataset = Dataset(training_df, train_labels, dict_moa)
valid_dataset = Dataset(validation_df, validation_labels, dict_moa)
test_dataset = Dataset(test_df, test_labels, dict_moa)

# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)


yn_class_weights = True
class_weights = apply_class_weights(training_set, device)
# loss_function

if yn_class_weights:
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()


# Creating Architecture
units = 64
drop  = 0.7

seq_model = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(p = drop),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, num_classes))

# optimizer_algorithm
learning_rate = 1e-3
optimizer = torch.optim.Adam(seq_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)

from ols import OnlineLabelSmoothing
loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=num_classes, smoothing = 0.05 ).to(device=device)
#optimizer = torch.optim.Adam(seq_model.parameters(), lr = learning_rate)

def training_loop(n_epochs, optimizer, model, loss_fn_train, loss_fn, train_loader, valid_loader):
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
    early_stopper = EarlyStopper(patience=8, min_delta=0.0001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        loss_fn_train.train()
        for tprofiles, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(tprofiles)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn_train(outputs,torch.max(labels, 1)[1])
            # For L2 regularization
            l2_lambda = 0.000001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            # Update weights
            loss.backward()
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            #labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == torch.max(labels, 1)[1]).sum())
        loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        loss_fn_train.next_epoch()
    # return lists with loss, accuracy every epoch
        if early_stopper.early_stop(validation_loss = val_loss):             
                break
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
            loss = loss_fn(outputs,torch.max(labels, 1)[1])
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == torch.max(labels, 1)[1]).sum()) # saving best 
            all_labels.append(torch.max(labels, 1)[1])
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
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'weighted'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_models/pre_split/' + 'ChemStruc_least_loss_model'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss

#def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    '''
    # lists keep track of loss and accuracy for training and validation set
    #optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, max_epochs +1), desc = "Epoch", position = 0, leave = True):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for compounds, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            #print(f' Compounds {compounds}')
            #print(f' Labels {labels}')
            
            #print(labels)
            # Training Model
            outputs = model(compounds)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,labels)
            # Update weights
            loss.backward()
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            #print(f' loss: {loss.item()}')
            train_predicted = torch.argmax(outputs, 1)
            #print(f' train_predicted {train_predicted}')
            # NEW
            #labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss, epoch)
        val_accuracy = val_correct/val_total
        # printing results for epoch
        if epoch == 1 or epoch %2 == 0:
            print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch
'''

# In[494]:


#def validation_loop(model, loss_fn, valid_loader, best_val_loss, epoch):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
    '''
    '''
    loss_val = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for compounds, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(compounds)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        if best_val_loss > loss_val:
            best_val_loss = loss_val
            torch.save(
                {'epoch': epoch,
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val
            },  '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_models/pre_split/' + 'ChemStruc_least_loss_model'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss

'''
def test_loop(model, loss_fn, test_loader):
    '''
    Assessing trained model on test dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    test_loader: generator creating batches of test data
    '''
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
            model = model.to(device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(compounds)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,torch.max(labels, 1)[1])
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
#----------------------------------------------------- Training and validation ----------------------------------#
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = seq_model,
              loss_fn_train=loss_fn_train,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator)
#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(p = drop),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, num_classes))
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_models/pre_split/' + 'ChemStruc_least_loss_model')['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = seq_model,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#
str_save = 'CS_' +  file_name
val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, str_save, '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split') 
val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  str_save, '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split')


#-------------------------------- Writing interesting info into terminal ----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)


table = [["Time to Run Program", elapsed_time],
['Accuracy of Test Set', accuracy_score(all_labels, all_predictions)],
['F1 Score of Test Set', f1_score(all_labels, all_predictions, average='macro')]]
print(tabulate(table, tablefmt='fancy_grid'))

run = neptune.init_run(project='erik-everett-palm/Tomics-Models', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run['model'] = 'Chemical Structure'
#run["feat_selec/feat_sel"] = feat_sel
run["filename"] = file_name
run['parameters/normalize'] = 'None'
# run['parameters/class_weight'] = class_weight
run['parameters/learning_rate'] = learning_rate
run['parameters/loss_function'] = str(loss_function)
#run['parameters/use_variance_threshold'] = use_variance_threshold
#f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_models/pre_split/' + 'ChemStruc_least_loss_model')
run['metrics/f1_score'] = state["f1_score"]
run['metrics/accuracy'] = state["accuracy"]
run['metrics/loss'] = state["valid_loss"]
run['metrics/time'] = elapsed_time
run['metrics/epochs'] = num_epochs

run['metrics/test_f1'] = f1_score(all_labels, all_predictions, average='macro')
run['metrics/test_accuracy'] = accuracy_score(all_labels, all_predictions)

conf_matrix_and_class_report(all_labels, all_predictions, str_save, dict_moa)

# Upload plots
run["images/loss"].upload('/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split'+ '/' + 'loss_train_val_' + str_save + now  + '.png')
run["images/accuracy"].upload('/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split' +'/' + 'acc_train_val_' + str_save + now + '.png')
import matplotlib.image as mpimg
conf_img = mpimg.imread('Conf_matrix.png')
run["files/classification_info"].upload("class_info.txt")
run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)