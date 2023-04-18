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
import warnings


# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune

from Erik_alll_helper_functions import checking_veracity_of_data, dict_splitting_into_tensor, val_vs_train_loss, val_vs_train_accuracy, EarlyStopper
from Erik_alll_helper_functions import conf_matrix_and_class_report, program_elapsed_time, dict_splitting_into_tensor
from Erik_alll_helper_functions import apply_class_weights, set_parameter_requires_grad, LogScaler
from Erik_alll_helper_functions import pre_processing, save_tprofile_npy, acquire_npy, np_array_transform, splitting
from Erik_alll_helper_functions import accessing_correct_fold_csv_files, create_splits, smiles_to_array, create_terminal_table, upload_to_neptune
from Helper_Models import DeepInsight_Model, Chem_Dataset, Reducer_profiles

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter('error')


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
model_name = 'CS_DeepInsight'
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# download csvs with all the data pre split
#cyc_adr_file = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_adr/'
#train_filename = 'cyc_adr_clue_train_fold_0.csv'
#val_filename = 'cyc_adr_clue_val_fold_0.csv'
#test_filename = 'cyc_adr_clue_test_fold_0.csv'
#training_set, validation_set, test_set =  load_train_valid_data(cyc_adr_file, train_filename, val_filename, test_filename)



file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = 'False', 'False'
if re.search('hq', file_name):
    hq = 'True'
if re.search('8', file_name):
    dose = 'True'
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
variance_thresh = int(input("Variance threshold? (Options: 0 - 1.2): "))
normalize_c = input("Normalize? (Options: True, False): ")
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

test_data_lst = list(test_set['Compound_ID'])
train_data_lst = list(training_set['Compound_ID'])
valid_data_lst = list(validation_set['Compound_ID'])

# check to make sure the compound IDs do not overlap
inter1 = set(test_data_lst) & set(train_data_lst)
inter2 = set(test_data_lst) & set(valid_data_lst)
inter3 = set(train_data_lst) & set(valid_data_lst)
assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")


num_classes = len(training_set.moa.unique())


# split data into labels and inputs
training_df, train_labels = splitting(training_set)
validation_df, validation_labels = splitting(validation_set)
test_df, test_labels = splitting(test_set)



# make sure that the number of labels is equal to the number of inputs
assert len(training_df) == len(train_labels)
assert len(validation_df) == len(validation_labels)
assert len(test_df) == len(test_labels)


# ----------------------------------------- TSNE Image Transformation ---------------------------------------#

X_train = df_train_features.values
X_val = df_val_features.values
y_train = df_train_labels.values
y_val = df_val_labels.values
X_test = df_test_features.values
y_test = df_test_labels.values

# scaling transcriptomics profiles to prepare for TSNE
ln = LogScaler()
X_train_norm = ln.fit_transform(df_train_features.to_numpy().astype(float))
X_val_norm = ln.transform(df_val_features.to_numpy().astype(float))
X_test_norm = ln.transform(df_test_features.to_numpy().astype(float))


# Specifying the TSNE parameters:
distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    learning_rate='auto',
    perplexity=5,
    n_jobs=-1
)
# In[52]:

# Specifying the pixel size of the TSNE image
#pixel_size = (10, 10)
#pixel_size = (20, 20)
#pixel_size = (30, 30)
pixel_size = (50,50)
#pixel_size = (100,100)
#pixel_size = (224,224)

# Specifying the Image Transformer
it = ImageTransformer(
    feature_extractor=reducer, 
    pixels=pixel_size)


# fitting Image Transformer
print("Fitting Image Transformer...")
it.fit(df_train_features, y= df_train_labels["moa"], plot=True)
print("Transforming...")
X_train_img = it.transform(X_train_norm)
X_val_img = it.transform(X_val_norm)
X_test_img = it.transform(X_test_norm)



import warnings; 
warnings.simplefilter('ignore')


import torchvision.models as models



num_classes = len(np.unique(df_train_labels["moa"]))
# In[47]:

# Transforming TSNE images into tensors
preprocess = transforms.Compose([
    transforms.ToTensor()
])

DI_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
DI_val_tensor = torch.stack([preprocess(img) for img in X_val_img]).float()
DI_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
# -----------------------------------------Prepping Individual Models ---------------------#
# parameters for the dataloader
batch_size = 100
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'shuffle' : True,
         'prefetch_factor' : 1} 


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#device = torch.device('cpu')
print(f'Training on device {device}. ' )



# load individual models
print("Loading Pretrained Models...")
modelCS =  nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(p = 0.7),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, num_classes))
modelCS.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'CS_model')['model_state_dict'])
modelDI = DeepInsight_Model()
modelDI.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + 'DeepInsight')['model_state_dict'])

# -----------------------------------------Prepping Ensemble Model ---------------------#
class CS_DI_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, tprofiles_df, labels_CID, dict_moa, transform = None):
        self.compound_df = compound_df
        self.tprofiles_df = tprofiles_df
        self.labels_CID = labels_CID
        self.dict_moa = dict_moa
        self.transform = transform
    def __len__(self):
        return len(self.tprofiles_df)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        tprofile = self.tprofiles_df[idx]
        CID, label  = self.labels_CID.iloc[idx]
        assert len(CID) > 1, "No compound ID found"
        smile_string = self.compound_df["SMILES"][self.compound_df["Compound_ID"]== CID]      # returns smiles by using compound as keys
        # assert smile_string.shape[0] == 1, "More than one compound found that matches Compound ID"
        # problem with enantiomers
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
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return tprofile, compound_array.float(), label_tensor.float() # returns 
        



# Create datasets with relevant data and labels
training_dataset_CSDI = CS_DI_Dataset(training_set, DI_train_tensor, df_train_labels, dict_moa)
valid_dataset_CSDI = CS_DI_Dataset(validation_set, DI_val_tensor, df_val_labels, dict_moa)
test_dataset_CSDI = CS_DI_Dataset(test_set, DI_test_tensor, df_test_labels, dict_moa)

# create generator that randomly takes indices from the training set
training_generator = torch.utils.data.DataLoader(training_dataset_CSDI, **params)
validation_generator = torch.utils.data.DataLoader(valid_dataset_CSDI, **params)
test_generator = torch.utils.data.DataLoader(test_dataset_CSDI, **params)

# create a model combining both models
class CStructure_DI(nn.Module):
    def __init__(self, modelCS, modelDI):
        super(CStructure_DI, self).__init__()
        self.modelCS = modelCS
        self.modelDI = modelDI
        self.linear_layer1 = nn.Linear(int(10 + 10), 25)
        self.selu = nn.SELU()
        self.Dropout = nn.Dropout(p = 0.5)
        self.linear_layer2 = nn.Linear(25,10)
       
    def forward(self, x1in, x2in):
        #print(f' compound: {x1.size()}')
        #print(f' image: {x2.size()}')
        if x1in.shape[1] != 2048:
            raise ValueError("The input shape for the compound is not correct")
        x1 = self.modelCS(x1in)
        x2 = self.modelDI(x2in)
        x  = torch.cat((x1, x2), dim = 1)
        x = self.Dropout(self.selu(self.linear_layer1(x)))
        output = self.linear_layer2(x)
        return output

CStructure_DI = CStructure_DI(modelCS, modelDI)

# optimizer_algorithm
#cnn_optimizer = torch.optim.Adam(updated_model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
learning_rate = 1e-6
optimizer = torch.optim.Adam(CStructure_DI.parameters(), lr = learning_rate)
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
        for tprofiles, cmpds, labels in tqdm(train_loader, desc = "Training", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cmpds, tprofiles)
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
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, Accuracy: {val_accuracy} ')
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
        for tprofiles, cmpds, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Assessing outputs
            outputs = model(cmpds, tprofiles)
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
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
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
        for tprofiles, compounds, labels in tqdm(test_loader,
                                            desc = "Test Batches w/in Epoch",
                                              position = 0,
                                              leave = True):
            # Move to device MAY NOT BE NECESSARY
            model = model.to(device)
            tprofiles = tprofiles.to(device = device)
            compounds = compounds.to(device = device)
            labels = labels.to(device= device)
            # Assessing outputs
            outputs = model(compounds, tprofiles)
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
        for tprofiles, cmpds, labels in tqdm(train_loader, desc = "Training", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Training Model
            outputs = model(cmpds, tprofiles)
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
        print(f' {datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, Accuracy: {val_accuracy} ')
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
        for tprofiles, cmpds, labels in valid_loader:
            # Move to device MAY NOT BE NECESSARY
            tprofiles = tprofiles.to(device = device)
            labels = labels.to(device= device)
            cmpds = cmpds.to(device = device)
            # Assessing outputs
            outputs = model(cmpds, tprofiles)
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

set_parameter_requires_grad(CStructure_DI, feature_extracting = True)

learning_rate = 1e-4
optimizer = torch.optim.Adam(CStructure_DI.parameters(), lr = learning_rate)
# loss_function
if incl_class_weights == True:
    class_weights = apply_class_weights(training_set, device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
num_epochs_fs = 20

train_loss_per_epoch_fe, train_acc_per_epoch_fe, val_loss_per_epoch_fe, val_acc_per_epoch_fe, num_epochs = training_loop_fe(n_epochs = num_epochs_fs,
              optimizer = optimizer,
              model = CStructure_DI,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              device = device)

set_parameter_requires_grad(CStructure_DI, feature_extracting = False)
learning_rate = 0.5e-4
optimizer = torch.optim.Adam(CStructure_DI.parameters(), lr = learning_rate)
# loss_function
if incl_class_weights == True:
    class_weights = apply_class_weights(training_set, device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
#----------------------------------------------------- Training and validation ----------------------------------#
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = CStructure_DI,
              loss_fn = loss_function,
              train_loader=training_generator, 
              valid_loader=validation_generator,
              device = device)
#----------------------------------------- Assessing model on test data -----------------------------------------#
model_test = CStructure_DI
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
                    init_learning_rate = learning_rate,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    variance_thresh = variance_thresh,
                    pixel_size = pixel_size,
                    loss_fn_train = False)
