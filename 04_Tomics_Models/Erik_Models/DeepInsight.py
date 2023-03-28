#!/usr/bin/env python
# coding: utf-8

# In[16]:


from pyDeepInsight import ImageTransformer, Norm2Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


# In[42]:


#!/usr/bin/env python
# coding: utf-8

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
# Neptune
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve,log_loss, accuracy_score, f1_score, classification_report
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.ensemble import VotingClassifier
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime as dt
import time
import joblib

from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import precision_recall_curve,log_loss
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import os
import pandas as pd
import numpy as np
import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import re
nn._estimator_type = "classifier"

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import *

start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# ------------------------------------------------------------------------------------------------------------------------------
'''
file = input("Which file would you like to use? (Options: tian10, erik10, erik10_hq, erik10, erik10_hq_dos, erik10_dos, cyc_adr, cyc_dop):")
variance_threshold = input("What variance threshold would you like to use? (Options: 0 - 1.2):")
normalize = input("Would you like to normalize the data? (Options: True, False):")
'''  

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

npy_exists = False
save_npy = True

df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = npy_exists,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = save_npy,
        data_subset = file_name)
checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)

# save moa labels, remove compounds
df_train_labels = df_train_labels["moa"]
df_val_labels = df_val_labels["moa"]
df_test_labels = df_test_labels["moa"]

X_train = df_train_features.values
X_val = df_val_features.values
y_train = df_train_labels.values
y_val = df_val_labels.values
X_test = df_test_features.values
y_test = df_test_labels.values


ln = LogScaler()
X_train_norm = ln.fit_transform(df_train_features.to_numpy().astype(float))
X_val_norm = ln.transform(df_val_features.to_numpy().astype(float))
X_test_norm = ln.transform(df_test_features.to_numpy().astype(float))


# In[51]:
from umap import UMAP

distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    learning_rate='auto',
    perplexity=5,
    n_jobs=-1
)
'''
reducer = UMAP(
    n_components=2,
    random_state=456
)
'''
# In[52]:


#pixel_size = (10, 10)
#pixel_size = (20, 20)
#pixel_size = (30, 30)
pixel_size = (50,50)
#pixel_size = (100,100)
#pixel_size = (224,224)
it = ImageTransformer(
    feature_extractor=reducer, 
    pixels=pixel_size)



# In[53]:

# fitting Image Transformer
print("Fitting Image Transformer...")
it.fit(df_train_features, y= df_train_labels, plot=True)
print("Transforming...")
X_train_img = it.transform(X_train_norm)
X_val_img = it.transform(X_val_norm)
X_test_img = it.transform(X_test_norm)


# In[54]:



import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

import torchvision


# In[43]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torchvision.models as models

preprocess = transforms.Compose([
    transforms.ToTensor()
])



# In[48]:


X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
y_train_tensor = df_train_labels
X_val_tensor = torch.stack([preprocess(img) for img in X_val_img]).float()
y_val_tensor = df_val_labels

X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
y_test_tensor = df_test_labels

# In[49]:
class Reducer_profiles(torch.utils.data.Dataset):
    def __init__(self, X, y, dict_moa):
        self.X = X
        self.y = y
        self.dict_moa = dict_moa

    def __getitem__(self, index):
        label = self.dict_moa[self.y[index]]
        return self.X[index], torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)
    
net = torchvision.models.squeezenet1_1(pretrained=True, progress=True)
num_classes = len(df_train_labels)
net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), 
                              stride=(1,1))
'''
class DeepInsight_Model(nn.Module):
    def __init__(self, model):
        super(DeepInsight_Model, self).__init__()
        self.base_model = model

    def forward(self, x):
        x = self.base_model(x)
        return x
'''
'''
# 10 by 10
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(64*1*1, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(-1, 64*1*1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x
        '''
 # 20 by 20
'''
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=64*10*10, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
'''
'''
# 30 by 30
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=64*15*15, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 '''

# 50 x 50
#deep
'''
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=256*12*12, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
'''

# not deep   50x50
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=64*25*25, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# instantiate the model

'''
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 64 * 50 * 50)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


'''

        
#DI_model = DeepInsight_Model(net)     
DI_model = DeepInsight_Model()


batch_size = 200

trainset = Reducer_profiles(X_train_tensor, y_train_tensor, dict_moa)
train_generator = DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = Reducer_profiles(X_val_tensor, y_val_tensor, dict_moa)
valid_generator = DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = Reducer_profiles(X_test_tensor, y_test_tensor, dict_moa)
test_generator = DataLoader(testset, batch_size=batch_size, shuffle=False)



# In[50]:

# If applying class weights
yn_class_weights = True
class_weights = apply_class_weights(training_set, device)
# loss_function
if yn_class_weights:
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
max_epochs = 75
optimizer = optim.SGD(
    net.parameters(),
    lr=1e-04,
    momentum=0.8,
    weight_decay=1e-05
)



# In[52]:
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
    early_stopper = EarlyStopper(patience=30, min_delta=0.0001)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
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
            loss = loss_fn(outputs, torch.max(labels, 1)[1])
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
            },  '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_models' +'/' + 'DeepInsight'
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


#------------------------------   Calling functions --------------------------- #
train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
              optimizer = optimizer,
              model = DI_model,
              loss_fn = loss_function,
              train_loader= train_generator, 
              valid_loader= valid_generator)

#--------------------------------- Assessing model on test data ------------------------------#
updated_model_test = DI_model
updated_model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_models' +'/' + 'DeepInsight')['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = updated_model_test,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#---------------------------------------- Visual Assessment ---------------------------------# 

val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, 'DI', '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images') 
val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  'DI', '/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images')

# results_assessment(all_predictions, all_labels, moa_dict)

#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()

elapsed_time = program_elapsed_time(start, end)

table = [["Time to Run Program", elapsed_time],
['Accuracy of Test Set', accuracy_score(all_labels, all_predictions)],
['F1 Score of Test Set', f1_score(all_labels, all_predictions, average='macro')]]
print(tabulate(table, tablefmt='fancy_grid'))

run = neptune.init_run(project='erik-everett-palm/Tomics-Models', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run['model'] = "deepinsight"
#run["feat_selec/feat_sel"] = feat_sel
run["filename"] = file_name
run['parameters/normalize'] = normalize_c
run['parameters/class_weight'] = yn_class_weights
run['parameters/variance_threshold'] = variance_thresh
# run['parameters/learning_rate'] = learning_rate
run['parameters/loss_function'] = str(loss_function)
run['parameters/pixel_size'] = pixel_size
run['parameters/reducer'] = str(reducer)
#run['parameters/use_variance_threshold'] = use_variance_threshold
#f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_models' +'/' + 'DeepInsight')
run['metrics/f1_score'] = state["f1_score"]
run['metrics/accuracy'] = state["accuracy"]
run['metrics/loss'] = state["valid_loss"]
run['metrics/time'] = elapsed_time
run['metrics/epochs'] = num_epochs


run['metrics/test_f1'] = f1_score(all_labels, all_predictions, average='macro')
run['metrics/test_accuracy'] = accuracy_score(all_labels, all_predictions)

conf_matrix_and_class_report(all_labels, all_predictions, 'DI', dict_moa)

# Upload plots
run["images/loss"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'loss_train_val_' + 'DI' + now + '.png')
run["images/accuracy"].upload('/home/jovyan/Tomics-CP-Chem-MoA/02_CP_Models/saved_images' + '/' + 'acc_train_val_' + 'DI' + now + '.png') 
import matplotlib.image as mpimg
conf_img = mpimg.imread('Conf_matrix.png')
run["files/classification_info"].upload("class_info.txt")
run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)
