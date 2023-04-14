#!/usr/bin/env python
# coding: utf-8

# In[1]:
#

# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate
import albumentations as A
import random

# Torch
import torch
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False
torch.backends.cuda.max_memory_allocated = 4*1024*1024*1024

from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn

import seaborn as sns
import neptune.new as neptune


import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')

from Erik_alll_helper_functions import channel_5_numpy, accessing_correct_fold_csv_files, dict_splitting_into_tensor
from Erik_alll_helper_functions import splitting, apply_class_weights,  program_elapsed_time, val_vs_train_loss
from Erik_alll_helper_functions import EarlyStopper, create_splits, val_vs_train_accuracy, conf_matrix_and_class_report

# Image analysis packages
import albumentations as A 
import cv2           
#pip install --upgrade efficientnet-pytorch  
from efficientnet_pytorch import EfficientNet
import re     
'''Albumentations is a Python library forfast and flexible image augmentations. 
Albumentations efficiently implements a rich variety of image transform operations that are optimized
for performance, and does so while providing a concise, yet powerful image augmentation interface for 
different computer vision tasks, including object classification, segmentation, and detection. '''
# https://albumentations.ai/docs/getting_started/image_augmentation/

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 1000 # number of epochs we are going to run 
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_df, labels_df, dict_moa, transform=None, image_normalization=None):
        self.img_labels = labels_df
        # print(self.img_labels)
        self.paths_df = paths_df
        self.transform = transform
        self.im_norm  = image_normalization
        self.dict_moa = dict_moa

    def __len__(self):
        ''' The number of data points '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''Retreiving the image '''
        # ID = self.list_ID[idx]
        image = channel_5_numpy(self.paths_df, idx, self.im_norm) # extract image from csv using index
        #print(image)
        #print(f' return from function: {image}')
        label = self.img_labels[idx]          # extract calssification using index
        #print(label)
        #label = torch.tensor(label, dtype=torch.short)
        label_tensor = torch.from_numpy(self.dict_moa[label]) # convert label to tensor
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        #return image.float(), label.long()
        return image.float(), label_tensor.float()  

    
class MyRotationTransform:
    " Rotate by one of the given angles"
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        angle = random.choice(self.angle)
        return TF.rotate(image, angle)
rotation_transform = MyRotationTransform([0,90,180,270])

# on the fly data augmentation 
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.3), 
    rotation_transform,
    transforms.RandomVerticalFlip(0.3)])
'''
train_transforms = A.Compose(
    [A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.2),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.4),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.5),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.6),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.8),
    A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),])
valid_transforms = A.Compose([])
'''
paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

file_name = "erik10"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = 'False', 'False'
if re.search('hq', file_name):
    hq = 'True'
if re.search('8', file_name):
    dose = 'True'
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

# download dictionary which associates moa with a number
dict_moa = dict_splitting_into_tensor(training_set)
assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique())

# extract compound IDs
test_data_lst= list(test_set["Compound_ID"].unique())
train_data_lst= list(training_set["Compound_ID"].unique())
valid_data_lst= list(validation_set["Compound_ID"].unique())

# check to make sure the compound IDs do not overlapp between the training, validation and test sets
inter1 = set(test_data_lst) & set(train_data_lst)
inter2 = set(test_data_lst) & set(valid_data_lst)
inter3 = set(train_data_lst) & set(valid_data_lst)
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")

# extracting all the paths to the images where we have a Compound in the respective lists
training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)

# removing the compounds that have a moa class that contains a "|" as this is not supported by the model
training_df = training_df[training_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
validation_df = validation_df[validation_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)
test_df = test_df[test_df.moa.str.contains("|", regex = False, na = True) == False].reset_index(drop=True)

# Checking to see if the compounds after removing from paths_v1v2 are the same as the ones in the training, validation and test sets
# no loss should occur, but it does occur
#assert len(list(training_df.compound.unique())) == len(train_data_lst)
#assert len(list(validation_df.compound.unique())) == len(valid_data_lst)
#assert len(list(test_df.compound.unique())) == len(test_data_lst)

# checking to see that the unique moa classes are identical across training, validation and test set
assert set(training_df.moa.unique()) == set(validation_df.moa.unique()) == set(test_df.moa.unique())
num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_df)
validation_df, validation_labels = splitting(validation_df)
test_df, test_labels = splitting(test_df)

# showing that I have no GPUs
world_size = torch.cuda.device_count()
# print(world_size)

# importing data normalization pandas dataframe
pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')


batch_size = 15
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 
          
# shuffle isn't working

# Datasets
#partition = partition
#labels = labels


if using_cuda:
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
else:
    device = torch.device('cpu')
print(f'Training on device {device}. ' )

# Create datasets with relevant data and labels
training_dataset = Dataset(training_df, train_labels, dict_moa, transform = train_transforms, image_normalization= pd_image_norm)
valid_dataset = Dataset(validation_df, validation_labels, dict_moa, image_normalization= pd_image_norm)
test_dataset = Dataset(test_df, test_labels, dict_moa, image_normalization= pd_image_norm)

# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset, **params)
test_generator = torch.utils.data.DataLoader(test_dataset, **params)



#-------------------------- Creating MLP Architecture ------------------------------------------#

class image_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b1', include_top=False, in_channels = 5)
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.Linear_last = nn.Linear(1280, num_classes)
        # self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.dropout_1(self.base_model(x))
        out = out.view(-1, 1280)
        out = self.Linear_last(out)
        # out = self.softmax(out) # don't need softmax when using CrossEntropyLoss
        return out

updated_model = image_network()
#num_classes = len(set(train['classes'].tolist())) 
# torchinfo.summary(updated_model, (batch_size, 5, 256,256), col_names=["kernel_size", "output_size", "num_params"])


yn_class_weights = True
class_weights = apply_class_weights(training_set, device)
# loss_function

if yn_class_weights:
    #loss_function = torch.nn.CrossEntropyLoss(class_weights)
    loss_function = torch.nn.BCEWithLogitsLoss(weight=class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()

#from ols import OnlineLabelSmoothing
#loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=num_classes, smoothing = 0.05).to(device=device)
loss_fn_train = False
#------------------------ Class weights, optimizer, and loss function ---------------------------------#


# optimizer_algorithm
learning_rate = 0.1
cnn_optimizer = torch.optim.SGD(updated_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(cnn_optimizer, 
                        milestones=[8, 16, 22, 28, 32, 36], # List of epoch indices
                        gamma = 0.5)
# --------------------------Function to perform training, validation, testing, and assessment ------------------


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
    early_stopper = EarlyStopper(patience=10, min_delta=0.0001)
    model = model.to(device)
    optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        #loss_fn_train.train()
        for imgs, labels in tqdm(train_loader,
                                 desc = "Train Batches w/in Epoch",
                                position = 0,
                                leave = False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            imgs = imgs.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(imgs)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            #loss = loss_fn_train(outputs,torch.max(labels, 1)[1])
            #loss = loss_fn(outputs,torch.max(labels, 1)[1])
            loss = loss_fn(outputs, labels)
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
            #labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == torch.max(labels, 1)[1]).sum())
        #loss_fn_train.eval()
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
        #loss_fn_train.next_epoch()
        scheduler.step()
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
        for imgs, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            imgs = imgs.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(imgs)
            #probs = torch.nn.Softmax(outputs)
            #loss = loss_fn(outputs,torch.max(labels, 1)[1])
            loss = loss_fn(outputs, labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            # labels = torch.argmax(labels,1)
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
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CP_model'
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
        for cp_imgs, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = False):
            # Move to device MAY NOT BE NECESSARY
            model = model.to(device)
            cp_imgs = cp_imgs.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(cp_imgs)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
            #loss = loss_fn(outputs,torch.max(labels, 1)[1])
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


#------------------------------   Calling functions --------------------------- #
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = cnn_optimizer,
              model = updated_model,
              loss_fn_train= loss_fn_train,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator)

#--------------------------------- Assessing model on test data ------------------------------#
updated_model_test = image_network()
updated_model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CP_model')['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = updated_model_test,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#---------------------------------------- Visual Assessment ---------------------------------# 
str_all = 'CP_' + file_name
val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, str_all, '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images') 
val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  str_all, '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images')

# results_assessment(all_predictions, all_labels, moa_dict)

#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()

elapsed_time = program_elapsed_time(start, end)

table = [["Time to Run Program", elapsed_time],
['Accuracy of Test Set', accuracy_score(all_labels, all_predictions)],
['F1 Score of Test Set', f1_score(all_labels, all_predictions, average='macro')]]
print(tabulate(table, tablefmt='fancy_grid'))

run = neptune.init_run(project='erik-everett-palm/Tomics-Models', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run['model'] = "CP"
#run["feat_selec/feat_sel"] = feat_sel
run["filename"] = file_name
run['parameters/normalize'] = "mean and std"
run['parameters/class_weight'] = yn_class_weights
run['parameters/learning_rate'] = learning_rate
run['parameters/loss_function'] = str(loss_function)
#run['parameters/use_variance_threshold'] = use_variance_threshold
#f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CP_model')
run['metrics/f1_score'] = state["f1_score"]
run['metrics/accuracy'] = state["accuracy"]
run['metrics/loss'] = state["valid_loss"]
run['metrics/time'] = elapsed_time
run['metrics/epochs'] = num_epochs

run['metrics/test_f1'] = f1_score(all_labels, all_predictions, average='macro')
run['metrics/test_accuracy'] = accuracy_score(all_labels, all_predictions)

conf_matrix_and_class_report(all_labels, all_predictions, str_all, dict_moa)

# Upload plots
run["images/loss"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'loss_train_val_' + str_all + now + '.png')
run["images/accuracy"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'acc_train_val_' + str_all + now + '.png') 
import matplotlib.image as mpimg
conf_img = mpimg.imread('Conf_matrix.png')
run["files/classification_info"].upload("class_info.txt")
run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)
