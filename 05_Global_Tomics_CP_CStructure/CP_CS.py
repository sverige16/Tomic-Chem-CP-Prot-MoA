from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem


from pyDeepInsight import ImageTransformer, Norm2Scaler
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

from Erik_alll_helper_functions import checking_veracity_of_data, dict_splitting_into_tensor, val_vs_train_loss, val_vs_train_accuracy, EarlyStopper
from Erik_alll_helper_functions import conf_matrix_and_class_report, program_elapsed_time, dict_splitting_into_tensor
from Erik_alll_helper_functions import apply_class_weights, set_parameter_requires_grad, LogScaler, create_terminal_table, upload_to_neptune
from Erik_alll_helper_functions import pre_processing, save_tprofile_npy, acquire_npy, np_array_transform, splitting
from Erik_alll_helper_functions import accessing_correct_fold_csv_files, create_splits, smiles_to_array
from Helper_Models import DeepInsight_Model, Chem_Dataset, Reducer_profiles, MyRotationTransform
from efficientnet_pytorch import EfficientNet 



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

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import re


from Erik_alll_helper_functions import checking_veracity_of_data, dict_splitting_into_tensor, val_vs_train_loss, val_vs_train_accuracy, EarlyStopper
from Erik_alll_helper_functions import conf_matrix_and_class_report, program_elapsed_time, dict_splitting_into_tensor
from Erik_alll_helper_functions import apply_class_weights, set_parameter_requires_grad, LogScaler
from Erik_alll_helper_functions import pre_processing, channel_5_numpy_CID, channel_5_numpy, splitting
from Erik_alll_helper_functions import accessing_correct_fold_csv_files, create_splits, smiles_to_array
from Helper_Models import image_network, Chem_Dataset, Reducer_profiles

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
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
print("Begin Training")
model_name = 'CP_CS'
#---------------------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")


file_name = 'erik10_hq_8_12'
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = 'False', 'False'
if re.search('hq', file_name):
    hq = 'True'
if re.search('8', file_name):
    dose = 'True'
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
variance_thresh = 0
normalize_c = 'False'

#variance_thresh = int(input("Variance threshold? (Options: 0 - 1.2): "))
#normalize_c = input("Normalize? (Options: True, False): ")
if variance_thresh > 0 or normalize_c == 'True':
    npy_exists = False
    save_npy = False

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


# ----------------------------------------- Prepping Chemical Structure Data ---------------------------------------#
# download dictionary which associates moa with a tensor

dict_moa = dict_splitting_into_tensor(training_set)
assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique()), "Training, validation and test sets have different labels"
assert df_train_features.shape[0] == df_train_features.dropna().shape[0], "NaNs in training set"


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_set)
validation_df, validation_labels = splitting(validation_set)
test_df, test_labels = splitting(test_set)



# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)




num_classes = len(np.unique(df_train_labels["moa"]))

# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 12
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# load individual models
print("Loading Pretrained Models...")
modelCS = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(p = 0.7),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, num_classes))
modelCS.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CS_model')['model_state_dict'])
modelCP = image_network()
modelCP.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CP_model')['model_state_dict'])

# -----------------------------------------Prepping Ensemble Model ---------------------#
class CS_CP_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, paths_df,  labels_CID, dict_moa, transform = None, im_norm = None):
        self.compound_df = compound_df
        self.paths_df = paths_df
        self.labels_CID = labels_CID
        self.dict_moa = dict_moa
        self.transform = transform
        self.im_norm = im_norm
    
    def __len__(self):
        return len(self.labels_CID)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        CID, label  = self.labels_CID.iloc[idx] 
        image = channel_5_numpy(self.paths_df, idx, self.im_norm)
        smile_string = self.compound_df["SMILES"][self.compound_df["Compound_ID"]== CID] 
        if smile_string.shape[0] > 1:
            row_num = smile_string.shape[0]
            selection = int(np.random.randint(0,row_num) - 1)
            smile_string = smile_string.iloc[selection]
            #print("We have an enantiomer")
        if type(smile_string) == pd.Series:
            smile_string = smile_string.values[0]
        elif type(smile_string) == str:
            smile_string = smile_string
        else:
            print("We have a problem")
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        return image.float(), compound_array.float(), label_tensor.float() # returns 
        
# ---------------------------------------- Prepping Cell Painting Dataset ---------------------#
rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation for CP Images
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])

# extract compound IDs
test_data_lst= list(test_set["Compound_ID"].unique())
train_data_lst= list(training_set["Compound_ID"].unique())
valid_data_lst= list(validation_set["Compound_ID"].unique())

# check to make sure the compound IDs do not overlapp
inter1 = set(test_data_lst) & set(train_data_lst)
inter2 = set(test_data_lst) & set(valid_data_lst)
inter3 = set(train_data_lst) & set(valid_data_lst)
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")

# downloading paths to all of the images
paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

# 
training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)



# importing data normalization pandas dataframe
pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')

# --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
# Create datasets with relevant data and labels
training_dataset_CSCP = CS_CP_Dataset(training_set, paths_v1v2, df_train_labels, dict_moa,transform = train_transforms, im_norm= pd_image_norm)
valid_dataset_CSCP = CS_CP_Dataset(validation_set, paths_v1v2, df_val_labels, dict_moa, transform = train_transforms, im_norm= pd_image_norm)
test_dataset_CSCP = CS_CP_Dataset(test_set, paths_v1v2, df_test_labels, dict_moa, transform = train_transforms, im_norm= pd_image_norm)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSCP, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSCP, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSCP, **params)

# create a model combining both models
class CS_CP(nn.Module):
    def __init__(self, modelCP, modelCS):
        super(CS_CP, self).__init__()
        self.modelCP = modelCP
        self.modelCS = modelCS
        self.linear_layer1 = nn.Linear(int(10 + 10), 25)
        self.selu = nn.SELU()
        self.Dropout = nn.Dropout(p = 0.5)
        self.linear_layer2 = nn.Linear(25,10)
       
    def forward(self, x1, x2):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        x1 = self.modelCP(x1)
        x2 = self.modelCS(x2)
        x  = torch.cat((x1, x2), dim = 1)
        x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer2(x)
        return output

CS_CP = CS_CP(modelCP, modelCS)

# optimizer_algorithm
#cnn_optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
learning_rate = 1e-6
optimizer = torch.optim.Adam(CS_CP.parameters(), lr = learning_rate)
# loss_function
if incl_class_weights == True:
    class_weights = apply_class_weights(training_set, device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()

# --------------------------------- Training, Test, Validation, Loops --------------------------------#
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, device):
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
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for cp_imgs, cmpds, labels in train_loader:
            optimizer.zero_grad()
            # put model, images, labels on the same device
            cp_imgs = cp_imgs.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cp_imgs, cmpds)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,torch.max(labels, 1)[1])
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
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss, device)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss},  Accuracy: {val_accuracy} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
    # return lists with loss, accuracy every epoch
        if early_stopper.early_stop(validation_loss = val_loss):             
                break
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                                

def validation_loop(model, loss_fn, valid_loader, best_val_loss, device):
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
        for cp_imgs , cmpds, labels in valid_loader:
            # put model, images, labels on the same device
            cp_imgs = cp_imgs.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cp_imgs, cmpds)
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
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + + model_name + ".pt"
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss


def test_loop(model, loss_fn, test_loader, device):
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
        for cp_img, cmpds, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = False):
            # put model, images, labels on the same device
            cp_img = cp_img.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cp_img, cmpds)
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


def training_loop_fe(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, device):
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
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for cp_imgs , cmpds, labels in train_loader:
            # put model, images, labels on the same device
            cp_imgs = cp_imgs.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cp_imgs, cmpds)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            loss = loss_fn(outputs,torch.max(labels, 1)[1])
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
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop_fe(model, loss_fn, valid_loader, best_val_loss, device)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
    # return lists with loss, accuracy every epoch
       
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, epoch
                                

def validation_loop_fe(model, loss_fn, valid_loader, best_val_loss, device):
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
        for cp_imgs, cmpds, labels in valid_loader:
            # put model, images, labels on the same device
            cp_imgs = cp_imgs.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cp_imgs, cmpds)
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
    model.train()
    return correct, total, avg_val_loss, best_val_loss

set_parameter_requires_grad(CS_CP, feature_extracting = True)

learning_rate = 1e-6
optimizer = torch.optim.Adam(CS_CP.parameters(), lr = learning_rate)
# loss_function
if incl_class_weights == True:
    class_weights = apply_class_weights(training_set, device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
num_epochs_fs = 20

train_loss_per_epoch_fe, train_acc_per_epoch_fe, val_loss_per_epoch_fe, val_acc_per_epoch_fe, num_epochs = training_loop_fe(n_epochs = num_epochs_fs,
              optimizer = optimizer,
              model = CS_CP,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              device = device)

set_parameter_requires_grad(CS_CP, feature_extracting = False)
learning_rate = 0.5e-6
optimizer = torch.optim.Adam(CS_CP.parameters(), lr = learning_rate)
# loss_function
if incl_class_weights == True:
    class_weights = apply_class_weights(training_set, device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()

#----------------------------------------------------- Training and validation ----------------------------------#
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = CS_CP,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              device = device)
#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CS_CP
model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt")['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = model_test,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator,
                                          device = device)

# ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/saved_images')


#-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = normalize_c,
                    yn_class_weights = incl_class_weights,
                    learning_rate = learning_rate, 
                    elapsed_time = elapsed_time, 
                    num_epochs = num_epochs,
                    loss_fn = loss_function,
                    all_labels = all_labels,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    variance_thresh = variance_thresh,
                    pixel_size = 256,
                    loss_fn_train = "false")
