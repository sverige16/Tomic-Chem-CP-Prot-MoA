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


import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

import torchvision

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
from Erik_alll_helper_functions import apply_class_weights, accessing_correct_fold_csv_files, create_splits
from Erik_alll_helper_functions import checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss
from Erik_alll_helper_functions import val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import pre_processing, create_terminal_table, upload_to_neptune

start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = "DeepInsight"
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
file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
hq, dose = 'False', 'False'
if re.search('hq', file_name):
    hq = 'True'
if re.search('8', file_name):
    dose = 'True'
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
#variance_thresh = int(input("Variance threshold? (Options: 0 - 1.2): "))
#normalize_c = input("Normalize? (Options: True, False): ")
variance_thresh = 0
normalize_c = 'False'
if variance_thresh > 0 or normalize_c == 'True':
    npy_exists = False
    save_npy = False
else:
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
#from umap import UMAP

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

# fitting Image Transformer
print("Fitting Image Transformer...")
it.fit(df_train_features, y= df_train_labels, plot=True)
print("Transforming...")
X_train_img = it.transform(X_train_norm)
X_val_img = it.transform(X_val_norm)
X_test_img = it.transform(X_test_norm)
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision.models as models

preprocess = transforms.Compose([
    transforms.ToTensor()
])


X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
y_train_tensor = df_train_labels
X_val_tensor = torch.stack([preprocess(img) for img in X_val_img]).float()
y_val_tensor = df_val_labels
X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
y_test_tensor = df_test_labels

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


batch_size = 50

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
'''
# loss_function 
ols_smooth = False
CEL = True
if ols_smooth:
    from ols import OnlineLabelSmoothing
    loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=num_classes, smoothing = 0.05).to(device=device)
    loss_function = torch.nn.CrossEntropyLoss(weight = class_weights)
else:
    loss_fn_train = False
    if CEL == True:
        loss_function = torch.nn.CrossEntropyLoss(weight = class_weights)
    else:
        loss_function = torch.nn.BCEWithLogitsLoss(weight = class_weights)
if yn_class_weights:
    #loss_function = torch.nn.CrossEntropyLoss(weight = class_weights) 
    loss_function = torch.nn.BCEWithLogitsLoss(weight = class_weights)
'''
from ols import OnlineLabelSmoothing
loss_fn = torch.nn.BCEWithLogitsLoss(weight = class_weights)
loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=num_classes, smoothing = 0.01).to(device=device)

#else:
 #   loss_function = torch.nn.CrossEntropyLoss()
#loss_function = torch.nn.CrossEntropyLoss(weight = class_weights)
#loss_function = torch.nn.BCEWithLogitsLoss(weight = class_weights)

max_epochs = 1000
learning_rate = 1e-04
optimizer = optim.SGD(
    DI_model.parameters(),
    lr=learning_rate,
    momentum=0.8,
    weight_decay=1e-05
)



# In[52]:
# --------------------------Function to perform training, validation, testing, and assessment ------------------


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, loss_fn_train = 'false'):
    '''
    n_epochs: number of epochs 
    optimizer: optimizer used to do backpropagation
    model: deep learning architecture
    loss_fn: loss function
    train_loader: generator creating batches of training data
    valid_loader: generator creating batches of validation data
    '''
    # lists keep track of loss and accuracy for training and validation set
    early_stopper = EarlyStopper(patience=15, min_delta=0.0001)
    model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(),weight_decay = 1e-6, lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    best_val_loss = np.inf
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        if loss_fn_train != "false":
            loss_fn_train.train()
        #loss_fn_train.train()
        for imgs, labels in tqdm(train_loader,
                                 desc = "Train Batches w/in Epoch",
                                position = 0,
                                leave = False):
            optimizer.zero_grad()
            #labels = label_smoothing(labels, 0.1, 10)
            # put model, images, labels on the same device
            imgs = imgs.to(device = device)
            labels = labels.to(device= device)
            # Training Model
            outputs = model(imgs)
            #print(f' Outputs : {outputs}') # tensor with 10 elements
            #print(f' Labels : {labels}') # tensor that is a number
            if loss_fn_train != "false":
                loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
       
            #loss = loss_fn(outputs, labels)
            
            #loss = l   oss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs, labels)

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
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        #loss_fn_train.eval()
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
    # return lists with loss, accuracy every epoch
        if early_stopper.early_stop(validation_loss = val_loss):             
            break
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
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
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #if ols_smooth or CEL:
          
            #else:
            loss = loss_fn(outputs, labels)
            #loss = loss_fn(outputs, labels)
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
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + ".pt"
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
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_uniform("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))

    class_weights = apply_class_weights(training_set, device)
    loss_fn = torch.nn.CrossEntropyLoss(class_weights)

    from ols import OnlineLabelSmoothing
    loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3)).to(device=device)

    max_epochs = 1000

    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, num_epochs = training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = scheduler)
    
  

    lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return (lowest1 + lowest2)/2

study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = num_classes, 
                                      training_generator= training_generator, 
                                      validation_generator = validation_generator), 
                                      n_trials=50)

#-------------------------------- Writing interesting info into terminal ------------------------# 

print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/" + model_name + '_' + now +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()
