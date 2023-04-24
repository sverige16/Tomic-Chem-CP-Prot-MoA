import torch.nn as nn
import pickle 
import numpy as np


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

from datetime import datetime
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
from torchsummary import summary
nn._estimator_type = "classifier"
import sys
import optuna
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (apply_class_weights, accessing_correct_fold_csv_files, create_splits, extract_tprofile,
                                        checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss,
                                        val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report,
                                        pre_processing, create_terminal_table, upload_to_neptune, dict_splitting_into_tensor,
                                        tprofiles_gc_too_func, set_bool_npy, set_bool_hqdose)

import numpy as np
import cv2
import os
start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

def differentiable_f1_loss(y_true, y_pred, class_weights=None):
    smooth = 1e-7

    # Convert predictions to binary by rounding
    y_pred = torch.round(torch.clamp(y_pred, 0, 1))

    # Calculate true positives, false positives, and false negatives
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    # Calculate continuous precision and recall
    p = tp / (tp + fp + smooth)
    r = tp / (tp + fn + smooth)

    # Calculate the differentiable F1 score
    f1 = 2 * p * r / (p + r + smooth)
    f1[torch.isnan(f1)] = 0

    # Apply class weights if provided
    if class_weights is not None:
        f1 = f1 * class_weights
        f1 /= torch.sum(class_weights)

    # Calculate the loss as 1 - F1 score
    return 1 - torch.mean(f1)

file_name = "erik10_hq_8_12"
def load_images(folder_path, train_range, test_range, validation_range):
    def read_image(file_path):
        img = cv2.imread(file_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def read_images_in_range(r):
        images = []
        for i in r:
            file_name = os.path.join(folder_path, f"{i}.png")
            if os.path.exists(file_name):
                images.append(read_image(file_name))
            else:
                print(f"File {file_name} not found.")
        return np.array(images)

    train_set = read_images_in_range(train_range)
    test_set = read_images_in_range(test_range)
    validation_set = read_images_in_range(validation_range)

    return train_set, test_set, validation_set

with open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/" + file_name + "labels_moa_dict" +'_' + 'fold0'+ ".pkl", 'rb') as f:
    all_labels = pickle.load(f)
train_labels, valid_labels, test_labels, dict_moa, dict_indexes = all_labels


# checking that the number of images and labels are the same
# Example usage:
folder_path = '/scratch2-shared/erikep/DWTM/ImageDataset/'
train_range = range(dict_indexes["train"][0], dict_indexes["train"][1]+1)  # Images 1-100 for training
validation_range = range(dict_indexes["valid"][0],dict_indexes["valid"][1]+1)  # Images 101-125 for testing
test_range = range(dict_indexes["test"][0], dict_indexes["test"][1]+1)  # Images 126-150 for validation

train_images, test_images, valid_images = load_images(folder_path, train_range, test_range, validation_range)
assert len(train_images) == len(train_labels)
assert len(valid_images) == len(valid_labels)
assert len(test_images) == len(test_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DWTM_profiles(torch.utils.data.Dataset):
    def __init__(self, feat_img, labels, dict_moa, transform = None):
        self.X = feat_img
        self.y = labels
        self.dict_moa = dict_moa
        self.transform = transform

    def __getitem__(self, index):
        label = dict_moa[self.y[index]]
        img = torch.tensor([self.X[index]], dtype=torch.float)
        if self.transform:
            img = self.transform(img)
        img = img.reshape(3, 128, 128)
        return img, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)

#DI_model = DeepInsight_Model(net)     

model_name = 'DWTM'
DWTM_model = models.resnet18(pretrained=False)
num_ftrs = DWTM_model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
DWTM_model.fc = nn.Linear(num_ftrs, len(dict_moa))
DWTM_model = DWTM_model.to(device)



batch_size = 50
train_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma = (0.1, 0.2))

trainset = DWTM_profiles(train_images, train_labels["moa"], dict_moa)
train_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = DWTM_profiles(valid_images, valid_labels["moa"], dict_moa)
valid_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = DWTM_profiles(test_images, test_labels["moa"], dict_moa)
test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# If applying class weights
yn_class_weights = True
if yn_class_weights:     # if we want to apply class weights
    class_weights = apply_class_weights(train_labels, device)
# loss_function
if yn_class_weights:
    #loss_function = torch.nn.CrossEntropyLoss(class_weights)
    #loss_function = loss_fn = torch.nn.BCEWithLogitsLoss(class_weights)
    from ols import OnlineLabelSmoothing
    loss_fn_train = "false"
    #loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=10, smoothing = 0.01).to(device=device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
max_epochs = 80
learning_rate = 1
optimizer = torch.optim.SGD(
    DWTM_model.parameters(),
    lr=learning_rate,
    momentum=0.8,
    weight_decay=1e-05
)


from torchvision import datasets, models, transforms





# In[52]:
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
    early_stopper = EarlyStopper(patience=25, min_delta=0.0001)
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
        #loss_fn_train.train()
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
            #loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
            loss = differentiable_f1_loss(outputs, labels, class_weights)
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
        # loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, best_val_loss_upd = validation_loop(model, loss_fn, valid_loader, best_val_loss)
        best_val_loss = best_val_loss_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' {datetime.now()} Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
    # return lists with loss, accuracy every epoch
        if early_stopper.early_stop(validation_loss = val_loss):             
                break
        #loss_fn_train.next_epoch()
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
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs, labels)
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            loss = differentiable_f1_loss(outputs, labels, class_weights)
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
            }, '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
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
            #loss = loss_fn(outputs, labels)
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
              model = DWTM_model,
              loss_fn_train= loss_fn_train,
              loss_fn = loss_function,
              train_loader= train_generator, 
              valid_loader= valid_generator)

#--------------------------------- Assessing model on test data ------------------------------#
updated_model_test = DWTM_model
updated_model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')['model_state_dict'])
correct, total, avg_test_loss, all_predictions, all_labels = test_loop(model = updated_model_test,
                                          loss_fn = loss_function, 
                                          test_loader = test_generator)

#---------------------------------------- Visual Assessment ---------------------------------# 
val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images') 
val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images')

# results_assessment(all_predictions, all_labels, moa_dict)

#-------------------------------- Writing interesting info into terminal ------------------------# 

end = time.time()

elapsed_time = program_elapsed_time(start, end)

create_terminal_table(elapsed_time, all_labels, all_predictions)
upload_to_neptune('erik-everett-palm/Tomics-Models',
                    file_name = file_name,
                    model_name = model_name,
                    normalize = "False",
                    yn_class_weights = yn_class_weights,
                    learning_rate = learning_rate, 
                    elapsed_time = elapsed_time, 
                    num_epochs = num_epochs,
                    loss_fn = loss_function,
                    all_labels = all_labels,
                    all_predictions = all_predictions,
                    dict_moa = dict_moa,
                    val_vs_train_loss_path = val_vs_train_loss_path,
                    val_vs_train_acc_path = val_vs_train_acc_path,
                    loss_fn_train = loss_fn_train)
