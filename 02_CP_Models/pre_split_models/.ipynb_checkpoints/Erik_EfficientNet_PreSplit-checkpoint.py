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
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import torch.nn as nn

import seaborn as sns
import neptune.new as neptune



from Erik_helper_functions import load_train_valid_data, dict_splitting_into_tensor, val_vs_train_loss, val_vs_train_accuracy, EarlyStopper
from Erik_helper_functions import  conf_matrix_and_class_report, program_elapsed_time


# Image analysis packages
import albumentations as A 
import cv2           
#pip install --upgrade efficientnet-pytorch  
from efficientnet_pytorch import EfficientNet     
'''Albumentations is a Python library forfast and flexible image augmentations. 
Albumentations efficiently implements a rich variety of image transform operations that are optimized
for performance, and does so while providing a concise, yet powerful image augmentation interface for 
different computer vision tasks, including object classification, segmentation, and detection. '''
# https://albumentations.ai/docs/getting_started/image_augmentation/

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 90 # number of epochs we are going to run 
apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

def splitting_into_tensor(df, num_classes, moa_dict):
    '''Splitting data into two parts:
    1. input : the pointer showing where the transcriptomic profile is  
    2. target one hot : labels (the correct MoA) '''
    
    # one-hot encoding labels
     # creating tensor from all_data.df
    for i in moa_dict.items():
        df['moa'] = df['moa'].replace(i[0], i[1])
    target = torch.tensor(df['moa'].values.astype(np.int64))

    # For each row, take the index of the target label
    # (which coincides with the score in our case) and use it as the column index to set the value 1.0.” 
    #target_onehot = torch.zeros(target.shape[0], num_classes)
    #target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    
    input =  df.drop('moa', axis = 1)
    
    return input, target #target_onehot

def image_normalization(image, channel, plate, pd_image_norm):
    '''
    Normalizes the image by the mean and standard deviation 
    Pseudocode:
    1. using plate and channel, extract mean and standard deviation from pd_imgnorm
    2. use torch.transform.Normalize to normalize the image
    3. return normalized image
    '''

    if channel == "C1":
        extract = plate
    elif channel == "C2":
        extract = plate + '.1'
    elif channel == "C3":
        extract = plate + '.2'
    elif channel == "C4":
        extract = plate + '.3'
    else:
        extract = plate + '.4'
    single_cha = pd_image_norm[extract]
    
    mean = float(single_cha.iloc[1])
    std = float(single_cha.iloc[2])
    im_np =  (image - mean) / std
    return im_np

def channel_5_numpy(df, idx, pd_image_norm):
    '''
    Puts together all channels from CP imaging into a single 5 x 256 x 256 tensor (c x h x w) from all_data.csv
    Input
    df  : file which contains all rows of image data with compound information (type = csv)
    idx : the index of the row (type = integer)
    
    Output:
    image: a single 5 x 256 x 256 tensor (c x h x w)
    '''
    # extract row with index 
    row = df.iloc[idx]
    
    # loop through all of the channels and add to single array
    im_list = []
    for c in range(1, 6):
        # extract by adding C to the integer we are looping
        #row_channel_path = row["C" + str(c)]
        local_im = cv2.imread(row["C" + str(c)], -1) # row.path would be same for me, except str(row[path]))
        
        # directly resize down to 256 by 256
        local_im = cv2.resize(local_im, (256, 256), interpolation = cv2.INTER_LINEAR)
        local_im = local_im.astype(np.float32)
        local_im_norm = image_normalization(local_im, c, row['plate'], pd_image_norm)
        # adds to array to the image vector 
        im_list.append(local_im_norm)
    
    arr_stack = np.stack(im_list, axis=0)
    # once we have all the channels, we covert it to a np.array, transpose so it has the correct dimensions and change the type for some reason
    #im = np.array(im).astype("int16")
    five_chan_img = torch.from_numpy(arr_stack)
    return five_chan_img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths_df, labels, transform=None, image_normalization=None):
        self.img_labels = labels
        # print(self.img_labels)
        self.paths_df = paths_df
        self.transform = transform
        self.im_norm  = image_normalization

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
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        #return image.float(), label.long()
        return image.float(), label.long()  

    
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

# testing using pandas dataframe
training_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_training_set_cyclo_adr_2.csv')
validation_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_valid_set_cyclo_adr_2.csv')
test_set = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/CS_test_set_cyclo_adr_2.csv')

# download dictionary which associates moa with a number
with open('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/cyclo_adr_2_moa_dict.pickle', 'rb') as handle:
        moa_dict = pickle.load(handle)

assert training_set.moa.unique().all() == validation_set.moa.unique().all() == test_set.moa.unique().all()

# extract compound IDs
test_data_lst= list(test_set["Compound_ID"].unique())
train_data_lst= list(training_set["Compound_ID"].unique())
valid_data_lst= list(validation_set["Compound_ID"].unique())

training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting_into_tensor(training_df, num_classes, moa_dict)
validation_df, validation_labels = splitting_into_tensor(validation_df, num_classes, moa_dict)
test_df, test_labels = splitting_into_tensor(test_df, num_classes, moa_dict)

# showing that I have no GPUs
world_size = torch.cuda.device_count()
# print(world_size)

# importing data normalization pandas dataframe
pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')


batch_size = 12 
# parameters
params = {'batch_size' : 12,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 2} 
          
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
training_dataset = Dataset(training_df, train_labels, transform = train_transforms, image_normalization= pd_image_norm)
valid_dataset = Dataset(validation_df, validation_labels, image_normalization= pd_image_norm)
test_dataset = Dataset(test_df, test_labels, image_normalization= pd_image_norm)

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


# If applying class weights
apply_class_weights = True
if apply_class_weights:     # if we want to apply class weights
    counts = training_set.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    #print(counts)
    class_weights = []   # create list that will hold class weights
    for moa in training_set.moa.unique():       # for each moa   
        #print(moa)
        counts[moa]
        class_weights.append(counts[moa])  # add counts to class weights
    #print(len(class_weights))
    #print(class_weights)
    #print(type(class_weights))
    # class_weights = 1 / (class_weights / sum(class_weights)) # divide all class weights by total moas
    class_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device) # transform into tensor, put onto device
#print(class_weights)

#------------------------ Class weights, optimizer, and loss function ---------------------------------#


# optimizer_algorithm
cnn_optimizer = torch.optim.Adam(updated_model.parameters(), weight_decay = 0.01, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
# loss_function
if apply_class_weights:
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()


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
    early_stopper = EarlyStopper(patience=5, min_delta=0.0001)
    model = model.to(device)
    optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, max_epochs +1), desc = "Epoch", position=0, leave= True):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        for imgs, labels in tqdm(train_loader,
                                 desc = "Train Batches w/in Epoch",
                                position = 0,
                                leave = True):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            imgs = imgs.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(imgs)
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
            #labels = torch.argmax(labels,1)
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
            loss = loss_fn(outputs,labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            # labels = torch.argmax(labels,1)
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
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'weighted'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_models' +'/' + 'CP_least_loss_model'
            )
    model.train()
    return correct, total, avg_val_loss, best_val_loss                      
'''
def validation_loop(model, loss_fn, valid_loader, best_val_loss):
    model = model.to(device)
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for imgs, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            imgs = imgs.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(imgs)
            # print(f' Outputs : {outputs}') # tensor with 10 elements
            # print(f' Labels : {labels}') # tensor that is a number
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
                {
                    'model_state_dict' : model.state_dict(),
                    'valid_loss' : loss_val
            },  '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_models' +'/' + 'CP_least_loss_model'
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


#------------------------------   Calling functions --------------------------- #
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = cnn_optimizer,
              model = updated_model,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator)

#--------------------------------- Assessing model on test data ------------------------------#
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = updated_model,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#---------------------------------------- Visual Assessment ---------------------------------# 

val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, 'CP', '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images') 
val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  'CP', '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images')

# results_assessment(all_predictions, all_labels, moa_dict)

#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()

elapsed_time = program_elapsed_time(start, end)

test_set_acc = f' {round(correct/total*100, 2)} %'
table = [["Time to Run Program", elapsed_time],
['Accuracy of Test Set', test_set_acc]]
print(tabulate(table, tablefmt='fancy_grid'))

run = neptune.init_run(project='erik-everett-palm/Tomics-Models', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run['model'] = "CNN"
#run["feat_selec/feat_sel"] = feat_sel
run["filename"] = "Cell_Painting_CNN"
run['parameters/normalize'] = "mean and std"
# run['parameters/class_weight'] = class_weight
# run['parameters/learning_rate'] = learning_rate
run['parameters/loss_function'] = str(loss_function)
#run['parameters/use_variance_threshold'] = use_variance_threshold
#f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_models' +'/' + 'CP_least_loss_model')
run['metrics/f1_score'] = state["f1_score"]
run['metrics/accuracy'] = state["accuracy"]
run['metrics/loss'] = state["valid_loss"]
run['metrics/time'] = elapsed_time
# run['metrics/epochs'] = max_epochs

conf_matrix_and_class_report(state["labels_val"], state["predictions"])

# Upload plots
run["images/loss"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'loss_train_val_' + now + '.png')
run["images/accuracy"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'acc_train_val_' + now + '.png')
import matplotlib.image as mpimg
conf_img = mpimg.imread('Conf_matrix.png')
run["files/classification_info"].upload("class_info.txt")
run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)
