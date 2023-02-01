#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import torchinfo


# Image analysis packages
import albumentations as A 
import cv2             
from efficientnet_pytorch import EfficientNet     
'''Albumentations is a Python library forfast and flexible image augmentations. 
Albumentations efficiently implements a rich variety of image transform operations that are optimized
for performance, and does so while providing a concise, yet powerful image augmentation interface for 
different computer vision tasks, including object classification, segmentation, and detection. '''
# https://albumentations.ai/docs/getting_started/image_augmentation/

# ----------------------------------------- hyperparameters ---------------------------------------#
# Hyperparameters
testing = False # decides if we take a subset of the data
max_epochs = 75 # number of epochs we are going to run 
apply_class_weights = True # weight the classes based on number of compounds
using_cuda = True # to use available GPUs
world_size = torch.cuda.device_count()

#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
'''# 10 selected MoAs 
moas_to_use = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 
        'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']



# read the data 
all_data = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/all_data.csv')

# Limiting the data to the first 2000 rows
testing = False
if testing == True:
    all_data = all_data[0:2000]





# In[7]:


# drop random previous index in table above
all_data.drop(all_data.columns[[0,11, 12]], axis=1, inplace=True) # Check that reading the data worked 


# In[9]:


dictionary = {'ATPase inhibitor': 7, 'Aurora kinase inhibitor': 0,
 'HDAC inhibitor': 4, 'HSP inhibitor': 9, 'JAK inhibitor': 2, 'PARP inhibitor': 6,
 'protein synthesis inhibitor': 3, 'retinoid receptor agonist': 8,
 'topoisomerase inhibitor': 5, 'tubulin polymerization inhibitor': 1}


# In[10]:


num_classes = len(dictionary) 


# In[11]:


# change moa to classes 
all_data['classes'] = None
for i in range(all_data.shape[0]):
    all_data.iloc[i, 10] = dictionary[all_data.iloc[i, 9]]


# In[13]:


# get the compound-MoA pair 
compound_moa = all_data[['compound','moa']].drop_duplicates()



# In[15]:


# we see that some of the classes have very few members because we limited the data set
compound_moa.moa.value_counts()





# In[17]:


# creating tensor from all_data.df
target = torch.tensor(all_data['classes'].values.astype(np.int64))



# In[18]:


target_onehot = torch.zeros(target.shape[0], num_classes)
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)



# In[20]:


# Dubbel check that only four well-represented classes remain
compound_moa.moa.value_counts()


# In[22]:


# split dataset into test and training/validation sets (10-90 split)
compound_train_valid, compound_test, moa_train_valid, moa_test = train_test_split(
  compound_moa.compound, compound_moa.moa, test_size = 0.10, stratify = compound_moa.moa, 
  shuffle = True, random_state = 1)
''' 


'''# In[24]:


# Split data set into training and validation sets (1 to 9)
# Same as above, but does split of only training data into training and validation data (in order to take advantage of stratification parameter)
compound_train, compound_valid, moa_train, moa_valid = train_test_split(
  compound_train_valid, moa_train_valid, test_size = 10/90, stratify = moa_train_valid,
  shuffle = True, random_state = 62757)

# In[26]:


# get the train, valid and test set by for every compound in data set, if it is in train, valid or test, return all info from all_data in new df.
train = all_data[all_data['compound'].isin(compound_train)]
valid = all_data[all_data['compound'].isin(compound_valid)]
test  = all_data[all_data['compound'].isin(compound_test)]



# ## Preparing Images

# In[30]:


# create dictionary for parition
partition = {"train": [], "valid": [], "test": []}

# create lists with indexes in splits
tr_list = train.index.tolist()
va_list = valid.index.tolist()

# place index into correct bin
for index in all_data.index.tolist():
#for index in compound_moa.index.tolist():
    if index in tr_list:
        partition["train"] += [index]
    elif index in va_list:
        partition["valid"]   += [index]
    else:
        partition["test"]  += [index]


# In[33]:


# create dictionary for labels, using PyTorch
labels = {}
for index,compound in zip(all_data.index.tolist(), target_onehot):
    labels[index] = compound


# In[37]:
'''

def pre_processing(testing):
        # 10 selected MoAs 
    moas_to_use = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 
            'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']


    # read the data 
    all_data = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/all_data.csv')



    # Limiting the data to the first 2000 rows
    if testing:
        all_data = all_data[0:1000]


    # drop random previous index in table above
    all_data.drop(all_data.columns[[0,11, 12]], axis=1, inplace=True) # Check that reading the data worked 

    dictionary = {'ATPase inhibitor': 7, 'Aurora kinase inhibitor': 0,
    'HDAC inhibitor': 4, 'HSP inhibitor': 9, 'JAK inhibitor': 2, 'PARP inhibitor': 6,
    'protein synthesis inhibitor': 3, 'retinoid receptor agonist': 8,
    'topoisomerase inhibitor': 5, 'tubulin polymerization inhibitor': 1}

    num_classes = len(dictionary) 

    # change moa to classes 
    all_data['classes'] = None
    for i in range(all_data.shape[0]):
        all_data.iloc[i, 10] = dictionary[all_data.iloc[i, 9]]

    # get the compound-MoA pair 
    compound_moa = all_data[['compound','moa']].drop_duplicates()


    #---------------------------- one_hot_encoding -------------------------------#

    # creating tensor from all_data.df
    target = torch.tensor(all_data['classes'].values.astype(np.int64))

    # For each row, take the index of the target label
    # (which coincides with the score in our case) and use it as the column index to set the value 1.0.” 
    target_onehot = torch.zeros(target.shape[0], num_classes)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)


    # --------------------------- splitting into training, validation and test sets----------#
    # split dataset into test and training/validation sets (10-90 split)
    compound_train_valid, compound_test, moa_train_valid, moa_test = train_test_split(
    compound_moa.compound, compound_moa.moa, test_size = 0.10, stratify = compound_moa.moa, 
    shuffle = True, random_state = 1)


    # Split data set into training and validation sets (1 to 9)
    # Same as above, but does split of only training data into training and validation 
    # data (in order to take advantage of stratification parameter)
    compound_train, compound_valid, moa_train, moa_valid = train_test_split(
    compound_train_valid, moa_train_valid, test_size = 10/90, stratify = moa_train_valid,
    shuffle = True, random_state = 62757)


    # get the train, valid and test set by for every compound in data set,
    #  if it is in train, valid or test, return all info from all_data in new df.
    train = all_data[all_data['compound'].isin(compound_train)]
    valid = all_data[all_data['compound'].isin(compound_valid)]
    # test  = all_data[all_data['compound'].isin(compound_test)]



    # create dictionary for parition
    partition = {"train": [], "valid": [], "test": []}

    # create lists with indexes in splits
    tr_list = train.index.tolist()
    va_list = valid.index.tolist()

    # place index into correct bin
    for index in all_data.index.tolist():
    #for index in compound_moa.index.tolist():
        if index in tr_list:
            partition["train"] += [index]
        elif index in va_list:
            partition["valid"]   += [index]
        else:
            partition["test"]  += [index]

    # create dictionary for labels, using PyTorch
    labels = {}
    for index,compound in zip(all_data.index.tolist(), target_onehot):
        labels[index] = compound
    return num_classes, partition, labels, compound_moa, all_data, dictionary


num_classes, partition, labels, compound_moa, all_data, dictionary = pre_processing(testing)

# on the fly data augmentation 
import albumentations as A
train_transforms = A.Compose([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.2),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.4),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.5),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.6),
    A.OneOf([A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),],p = 0.8),
    A.Flip(),A.ShiftScaleRotate(scale_limit=0.2),A.RandomRotate90(),])
valid_transforms = A.Compose([])


def channel_5_numpy(df, idx):
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
    im = []
    for c in range(1, 6):
        # extract by adding C to the integer we are looping
        #row_channel_path = row["C" + str(c)]
        local_im = cv2.imread(row.path + "/" + row["C" + str(c)], -1) # row.path would be same for me, except str(row[path]))
        
        # directly resize down to 256 by 256
        local_im = cv2.resize(local_im, (256, 256), interpolation = cv2.INTER_LINEAR)
        # adds to array to the image vector 
        im.append(local_im)

    # once we have all the channels, we covert it to a np.array, transpose so it has the correct dimensions and change the type for some reason
    im = np.array(im).astype("int16")
    return torch.from_numpy(im)




class Dataset(torch.utils.data.Dataset):
    def __init__(self, partition, labels, transform=None):
        self.img_labels = labels
        # print(self.img_labels)
        self.list_ID = partition
        self.transform = transform

    def __len__(self):
        ''' The number of data points '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''Retreiving the image '''
        ID = idx
        # ID = self.list_ID[idx]
        image = channel_5_numpy(all_data, ID) # extract image from csv using index
        #print(f' return from function: {image}')
        label = self.img_labels[ID]          # extract calssification using index
        #print(label)
        #label = torch.tensor(label, dtype=torch.short)
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            image = self.transform(image)
        #return image.float(), label.long()
        return image.float(), label.float()



# showing that I have no GPUs
world_size = torch.cuda.device_count()
# print(world_size)



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

# Create a dataset with all indices and labels
everything = Dataset(partition["train"]+partition['valid']+partition['test'], labels)

#------------------------------------- Generators --------------------------------#
# generator: training
# create a subset with only train indices
training_set = torch.utils.data.Subset(everything, partition["train"])

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_set, **params)
# training_set = Dataset(partition["train"], labels)

valid_set = torch.utils.data.Subset(everything, partition["valid"])
valid_generator = torch.utils.data.DataLoader(valid_set, **params)

test_set = torch.utils.data.Subset(everything, partition["test"])
test_generator = torch.utils.data.DataLoader(test_set, **params)



# # Efficient Net

# In[55]:

         # have to download it, since latest version can use torchvision.models
#base_model = EfficientNet.from_name('efficientnet-b1', include_top=False, in_channels = 5)
#torchinfo.summary(base_model, (32, 5, 256,256), col_names=["kernel_size", "output_size", "num_params"])
#custom_model = nn.Sequential(model,View((2)), nn.Dropout(0.3))
#print(torchinfo.summary(custom_model, (32,5,256,256)))



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


# In[60]:


# If applying class weights
if apply_class_weights:     # if we want to apply class weights
    counts = compound_moa.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    #print(counts)
    class_weights = []   # create list that will hold class weights
    for moa in compound_moa.moa.unique():       # for each moa   
        #print(moa)
        counts[moa]
        class_weights.append(counts[moa])  # add counts to class weights
    #print(len(class_weights))
    #print(class_weights)
    #print(type(class_weights))
    # class_weights = 1 / (class_weights / sum(class_weights)) # divide all class weights by total moas
    class_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device) # transform into tensor, put onto device


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
        for imgs, labels in train_loader:
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
            labels = torch.argmax(labels,1)
            #print(labels)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        if epoch == 1 or epoch %5 == 0:
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
            labels = torch.argmax(labels,1)
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

def results_assessment(y_true, y_pred):
    save_path = '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/'
    target_names = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor',
     'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 'topoisomerase inhibitor', 
     'PARP inhibitor','ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']
    #print(f' true {y_true}')
    #print(f' prediction {y_pred}')
    for count, ele in enumerate(target_names):
        #print(count)
        #print(ele)
        #print(dictionary[ele])
        assert count == dictionary[ele], "Mismatch with dictionary"
    
    class_report_output = classification_report(y_true, y_pred)
    try:
        chem_struc_file= open(save_path + 'saved_classification_reports' + '/' + now + '_classif_report.txt', 'a')
        chem_struc_file.write((class_report_output))
        chem_struc_file.close()
    except:
        print("Unable to append to file")
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    np.save((save_path + 'saved_confusion_matrices/'  + now + '_confusion_matrix.npy'), conf_matrix)



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
            labels = torch.argmax(labels,1)
            #print(predicted)
            #print(labels)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
            #print(f' Predicted: {predicted.tolist()}')
            #print(f' Labels: {predicted.tolist()}')
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + labels.tolist()
        results_assessment(all_predictions, all_labels)
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss

#---------------------------------------- Visual Assessment ---------------------------------# 

def val_vs_train_loss(epochs, train_loss, val_loss):
    ''' 
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_loss: training loss per epoch (python list)
    val_loss: validation loss per epoch (python list)
    ''' 
    loss_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images'
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
    acc_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images'
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

#------------------------------   Calling functions --------------------------- #
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch = training_loop(n_epochs = max_epochs,
              optimizer = cnn_optimizer,
              model = updated_model,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=valid_generator)



val_vs_train_loss(max_epochs,train_loss_per_epoch, val_loss_per_epoch)


val_vs_train_accuracy(max_epochs, train_acc_per_epoch, val_acc_per_epoch)

correct, total, avg_test_loss = test_loop(model = updated_model,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#-------------------------------- Writing interesting info into terminal ------------------------# 

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

