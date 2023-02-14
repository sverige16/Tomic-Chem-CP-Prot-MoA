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
# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 250 # number of epochs we are going to run 
apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
#---------------------------------------------------------------------------------------------------------------------------------------#
def splitting_into_tensor(df, num_classes):
    '''Splitting data into two parts:
    1. input : the pointer showing where the transcriptomic profile is  
    2. target one hot : labels (the correct MoA) '''
    
    # one-hot encoding labels
     # creating tensor from all_data.df
    target = torch.tensor(df['moa'].values.astype(np.int64))

    # For each row, take the index of the target label
    # (which coincides with the score in our case) and use it as the column index to set the value 1.0.” 
    #target_onehot = torch.zeros(target.shape[0], num_classes)
    #target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    
    input =  df.drop('moa', axis = 1)
    
    return input, target #target_onehot


# A function changing SMILES to Morgan fingerprints 
def smiles_to_array(smiles):
    molecules = Chem.MolFromSmiles(smiles) 
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(molecules, 2)
    x_array = []
    arrays = np.zeros(0,)
    DataStructs.ConvertToNumpyArray(fingerprints, arrays)
    x_array.append(arrays)
    x_array = np.asarray(x_array)
    x_array = ((np.squeeze(x_array)).astype(int)) 
    x_array = torch.from_numpy(x_array)
    return x_array                  

def val_vs_train_loss(epochs, train_loss, val_loss):
    ''' 
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_loss: training loss per epoch (python list)
    val_loss: validation loss per epoch (python list)
    ''' 
    loss_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split'
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_loss, label = "train_loss")
    plt.plot(x_axis, val_loss, label = "val_loss")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.title('Validation versus Training Loss: CP Image Model')
    plt.legend()
    # plot
    plt.savefig(loss_path_to_save + '/' + 'loss_train_val_' + now)

def val_vs_train_accuracy(epochs, train_acc, val_acc):
    '''
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_acc: accuracy loss per epoch (python list)
    val_acc: accuracy loss per epoch (python list)
    '''
    acc_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/saved_images/pre_split'
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_acc, label = "train_acc")
    plt.plot(x_axis, val_acc, label = "val_acc")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation versus Training Accuracy: CP Image Model')
    plt.legend()
    # plot
    plt.savefig(acc_path_to_save + '/' + 'acc_train_val_' + now)

def results_assessment(y_true, y_pred, dict_moa):
    target= [None]*len(dict_moa)
    for i in dict_moa.items():
        target[i[1]] = i[0]
    save_path =  '/home/jovyan/Tomics-CP-Chem-MoA/01_CStructure_Models/'
    class_report_output = classification_report(y_true, y_pred, target_names=target)
    try:
        chem_struc_file= open(save_path + 'saved_classification_reports/pre_split/' + now + '_classif_report.txt', 'a')
        chem_struc_file.write((class_report_output))
        chem_struc_file.close()
    except:
        print("Unable to append to file")
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    np.save((save_path + 'saved_confusion_matrices/pre_split/'  + now + '_confusion_matrix.npy'), conf_matrix)

   
#---------------------------------------------------------------------------------------------------------------------------------------#
# create Torch.dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, labels, transform=None):
        self.compound_labels = labels    # the entire length of the correct classes that we are trying to predict
        # print(self.img_labels)
        self.compound_df = compound_df        # list of indexes that are a part of training, validation, tes sets
        self.transform = transform       # any transformations done

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
        label = self.compound_labels[idx]             # extract classification using index
        #print(f' label: {label}')
        #label = torch.tensor(label, dtype=torch.float)
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return compound_array.float(), label.long()

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

# download csvs with all the data pre split
training_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_training_set_cyclo_adr_2.csv')
validation_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_valid_set_cyclo_adr_2.csv')
test_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_test_set_cyclo_adr_2.csv')

# download dictionary which associates moa with a number
with open('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/cyclo_adr_2_moa_dict.pickle', 'rb') as handle:
        moa_dict = pickle.load(handle)

assert training_set.moa.unique().all() == validation_set.moa.unique().all() == test_set.moa.unique().all()


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting_into_tensor(training_set, num_classes)
validation_df, validation_labels = splitting_into_tensor(validation_set, num_classes)
test_df, test_labels = splitting_into_tensor(test_set, num_classes)



batch_size = 50
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )


# Create datasets with relevant data and labels
training_dataset = Dataset(training_df, train_labels)
valid_dataset = Dataset(validation_df, validation_labels)
test_dataset = Dataset(test_df, test_labels)

# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)


# If applying class weights
apply_class_weights = True
if apply_class_weights:     # if we want to apply class weights
    counts = training_set.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    print(counts)
    class_weights = []   # create list that will hold class weights
    for moa in training_set.moa.unique():       # for each moa   
        #print(moa)
        counts[moa]
        class_weights.append(counts[moa])  # add counts to class weights
    print(len(class_weights))
    print(class_weights)
    print(type(class_weights))
    # class_weights = 1 / (class_weights / sum(class_weights)) # divide all class weights by total moas
    class_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device) # transform into tensor, put onto device
print(class_weights)


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
#cnn_optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
optimizer = torch.optim.Adam(seq_model.parameters(), lr = 1e-4)
# loss_function
if apply_class_weights == True:
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
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


# In[494]:


def validation_loop(model, loss_fn, valid_loader, best_val_loss, epoch):
    '''
    Assessing trained model on valiidation dataset 
    model: deep learning architecture getting updated by model
    loss_fn: loss function
    valid_loader: generator creating batches of validation data
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
            loss = loss_fn(outputs,labels)
            loss_test += loss.item()
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
            #print(f' Predicted: {predicted.tolist()}')
            #print(f' Labels: {predicted.tolist()}')
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + labels.tolist()
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss, all_predictions, all_labels
#----------------------------------------------------- Training and validation ----------------------------------#
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = seq_model,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator)
#----------------------------------------- Assessing model on test data -----------------------------------------#
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = seq_model,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#
val_vs_train_loss(max_epochs,train_loss_per_epoch, val_loss_per_epoch)

val_vs_train_accuracy(max_epochs, train_acc_per_epoch, val_acc_per_epoch)

results_assessment(all_predictions, all_labels, moa_dict)
#-------------------------------- Writing interesting info into terminal ----------------------------------# 
end = time.time()
def program_elapsed_time(start, end):
    program_time = round(end - start, 2) 
    print(program_time)
    if program_time > float(60) and program_time < 60*60:
        program_time =  program_time/60
        time_elapsed = str(program_time) + ' min'
    elif program_time > 60*60:
        program_time = program_time/3600
        time_elapsed = str(program_time) + ' hrs'
    else:
        time_elapsed = str(program_time) + ' sec'
    return time_elapsed
program_elapsed_time = program_elapsed_time(start, end)

test_set_acc = f' {round(correct/total*100, 2)} %'
table = [["Time to Run Program", program_elapsed_time],
['Accuracy of Test Set', test_set_acc]]
print(tabulate(table, tablefmt='fancy_grid'))
