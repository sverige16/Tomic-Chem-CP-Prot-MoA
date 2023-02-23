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

import optuna


# In[37]:


from ML_battery_L1000_dim_reduc import tprofiles_gc_too_func, extract_tprofile, load_train_valid_data 
from ML_battery_L1000_dim_reduc import variance_threshold, normalize_func, feature_selection


# In[63]:


using_cuda = True
hidden_size = 1000


# In[39]:


# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")


# In[40]:
def save_val(val_tensor, file_type):
    torch.save(val_tensor, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/pickles/val_order_pickles/' + file_type )
    print('Done writing binary file')


def dict_splitting_into_tensor(df):
    '''Splitting data into two parts:
    1. input : the pointer showing where the transcriptomic profile is  
    2. target one hot : labels (the correct MoA) '''
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df["moa"].unique().reshape(-1,1))
    one_hot_encoded = enc.transform(enc.categories_[0].reshape(-1,1)).toarray()
    dicti = {}
    for i in range(0, len(enc.categories_[0])):
        dicti[str(enc.categories_[0][i])] = one_hot_encoded[i]
    return dicti

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


# In[65]:


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


train_filename = 'L1000_training_set_nv_cyc_adr.csv'
valid_filename = 'L1000_test_set_nv_cyc_adr.csv'
L1000_training, L1000_validation = load_train_valid_data(train_filename, valid_filename)
# Creating a  dictionary of the one hot encoded labels
dict_moa = dict_splitting_into_tensor(L1000_training)

# shuffling training and validation data
L1000_training = L1000_training.sample(frac = 1, random_state = 1)
L1000_validation = L1000_validation.sample(frac = 1, random_state = 1)

print("extracting training transcriptomes")
profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
print("extracting validation transcriptomes")
profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)

# ------------------------------------- Feature Selection -------------------------------------#
# select features based on most useful features for linear models
#feature_selection()
# select features based on variance threshold

# ------------------------------------- Normalization -------------------------------------#
# normalize the data



num_classes = len(L1000_training["moa"].unique())



# In[49]:


# generator: training
# create a subset with only train indices

# create generator that randomly takes indices from the training set
training_dataset = Transcriptomic_Profiles(profiles_gc_too_train, L1000_training, dict_moa)



training_generator = torch.utils.data.DataLoader(training_dataset, **params)

# In[50]:


# generator: validation
# create a subset with only valid indices

# create generator that randomly takes indices from the validation set
validation_dataset = Transcriptomic_Profiles(profiles_gc_too_valid, L1000_validation, dict_moa)



validation_generator = torch.utils.data.DataLoader(validation_dataset, **params)





# In[67]:


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
    model = model.to(device)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, max_epochs +1), desc = "Epoch", position=0, leave= True):
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
            loss = loss_fn(outputs,labels)
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
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
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
            loss = loss_fn(outputs,labels)
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
                    'predictions' : pred_cpu,
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val,
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'weighted'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_models' +'/' + 'Tomics_SimpleNN'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss

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

    loss_fn = torch.nn.BCEWithLogitsLoss()
    max_epochs = 30


    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = model,
              loss_fn = loss_fn,
              train_loader=training_generator, 
              valid_loader=validation_generator)
    return min(val_loss_per_epoch)

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = 2, 
                                      training_generator= training_generator, 
                                      validation_generator = validation_generator), 
                                      n_trials=50)

#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()
def program_elapsed_time(start, end):
    program_time = round(end - start, 2) 
    if program_time > float(60) and program_time < 60*60:
        program_time =  program_time/60
        time_elapsed = str(program_time) + ' min'
    elif program_time > 60*60:
        program_time = program_time/3600
        time_elapsed = str(program_time) + ' hrs'
    else:
        time_elapsed = str(program_time) + ' sec'
    return time_elapsed
elapsed_time = program_elapsed_time(start, end)

#test_set_acc = f' {round(correct/total*100, 2)} %'
table = [["Time to Run Program", elapsed_time]]
#['Accuracy of Test Set', test_set_acc]]
print(tabulate(table, tablefmt='fancy_grid'))


run = neptune.init_run(project='erik-everett-palm/Tomics-Models', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run['model'] = 'Tomics_SimpleNN_optuna'
run["metric/best_loss"] = study.best_value
run["filename"] = train_filename
run['parameters/normalize'] = 'None'
run['parameters/best_params'] = str(study.best_params)
# run['parameters/class_weight'] = class_weight
run['parameters/learning_rate'] = study.best_params['lr']
run['parameters/loss_function'] = 'BCEWithLogitsLoss'
run['parameters/optimizer'] = study.best_params['optimizer']
run['parameters/epochs'] = max_epochs
run['parameters/n_layers'] = study.best_params['n_layers']

#run['parameters/use_variance_threshold'] = use_variance_threshold
#f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)

'''state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_models' +'/' + 'Tomics_SimpleNN')
run['metrics/f1_score'] = state["f1_score"]
run['metrics/accuracy'] = state["accuracy"]
run['metrics/loss'] = state["valid_loss"]
run['metrics/time'] = elapsed_time
run['metrics/epochs'] = max_epochs

# Upload plots
run["images/loss"].upload("/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images"+ '/' + 'loss_train_val_simpleNN' + now + '.png')
run["images/accuracy"].upload('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images' + '/' + 'acc_train_val_simpleNN' + now + '.png')
'''