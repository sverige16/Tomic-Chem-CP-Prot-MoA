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
from Erik_alll_helper_functions import apply_class_weights, accessing_correct_fold_csv_files, create_splits
from Erik_alll_helper_functions import checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss
from Erik_alll_helper_functions import val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report
from Erik_alll_helper_functions import pre_processing, create_terminal_table, upload_to_neptune, find_two_lowest_numbers



start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = "IGTD"
print("Begin Training")

# ------------ Load .pkl file ------------ #
# Euclidean distances with absolute error
'''
with open('/scratch2-shared/erikep/Results/Euc_full/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "euclid"

with open('/scratch2-shared/erikep/Results/Euc_full/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/Pear/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"

with open('/scratch2-shared/erikep/Results/Pear/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/hq_Pear/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"

with open('/scratch2-shared/erikep/Results/hq_Pear/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
'''
# pearson correlation with squared error
with open('/scratch2-shared/erikep/Results/erik10_hq_8_12_Pearson/Results_imag.pkl', 'rb') as f:
    data = pickle.load(f)
    data_set_name = "pearson"
    file_name = "erik_hq_8_12"
    variance_thresh = 0
    norm = False

with open('/scratch2-shared/erikep/Results//erik10_hq_8_12_Pearson/Results_samp.pkl', 'rb') as f:
    samples = pickle.load(f)
generated_images = np.transpose(data, (2, 0, 1))

# ------------- Load Labels -------------- #
with open('/scratch2-shared/erikep/Results/labels_hq_moadict.pkl', 'rb') as f:
    all_labels = pickle.load(f)
train_labels, valid_labels, test_labels, dict_moa, dict_indexes = all_labels

num_classes = len(dict_moa)

last_training_index = train_labels.shape[0]
last_validat_index = last_training_index + valid_labels.shape[0]
last_test_index = generated_images.shape[0]
# split generated images into train, test and validation
train_images = generated_images[:last_training_index]
valid_images = generated_images[last_training_index: last_validat_index]
test_images = generated_images[last_validat_index: last_test_index]
'''train_images = generated_images[:train_labels.shape[0]]
valid_images = pd.DataFrame(generated_images[(int(train_labels.shape[0])) + 1  : (int(train_labels.shape[0]) + 1  + int(valid_labels.shape[0]))]).reset_index(drop=True)
test_images = pd.DataFrame(generated_images[(int(train_labels.shape[0]) + int(valid_labels.shape[0])+ 1):train_labels.shape[0] + valid_labels.shape[0] + test_labels.shape[0]]).reset_index(drop=True)
'''

# checking that the number of images and labels are the same
assert len(train_images) == len(train_labels)
assert len(valid_images) == len(valid_labels)
assert len(test_images) == len(test_labels)

assert samples[last_training_index] == '0'
assert samples[last_validat_index] == '0'
#assert samples[last_test_index -1 ] ==  '2706'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IGTD_profiles(torch.utils.data.Dataset):
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
        return img, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)

class IGTD_Model(nn.Module):
    def __init__(self, channel_number, hidden_layer1, hidden_layer2, dropout_rate1):
        super(IGTD_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels= channel_number, kernel_size=(2,2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(2,2), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.dropout = nn.Dropout(dropout_rate1)
        self.fc1 = nn.Linear(16*4* 82, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, 10)
        
    def forward(self, x):
        #x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        #x = self.conv2_bn(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 16*4* 82
                   )
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x     
#DI_model = DeepInsight_Model(net)     
IGTD_model = IGTD_Model()
batch_size = 50
train_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma = (0.1, 0.2))

trainset = IGTD_profiles(train_images, train_labels["moa"], dict_moa)
train_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = IGTD_profiles(valid_images, valid_labels["moa"], dict_moa)
valid_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = IGTD_profiles(test_images, test_labels["moa"], dict_moa)
test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)



# In[50]:

# If applying class weights
yn_class_weights = True
if yn_class_weights:     # if we want to apply class weights
    class_weights = apply_class_weights(train_labels, device)
# loss_function
if yn_class_weights:
    #loss_function = torch.nn.CrossEntropyLoss(class_weights)
    #loss_function = loss_fn = torch.nn.BCEWithLogitsLoss(class_weights)
    from ols import OnlineLabelSmoothing
    loss_fn_train = OnlineLabelSmoothing(alpha = 0.5, n_classes=10, smoothing = 0.01).to(device=device)
    loss_function = torch.nn.CrossEntropyLoss(class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()
max_epochs = 80
learning_rate = 1e-04
optimizer = torch.optim.SGD(
    IGTD_model.parameters(),
    lr=learning_rate,
    momentum=0.8,
    weight_decay=1e-05
)

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
        loss_fn_train.train()
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
            loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
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
        loss_fn_train.eval()
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
            #probs = torch.nn.Softmax(outputs)
            #loss = loss_fn(outputs, torch.max(labels, 1)[1])
            #loss = loss_fn(outputs, labels)
            loss = loss_fn(outputs, torch.max(labels, 1)[1])
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


def objectiv(trial, num_feat, num_classes, training_generator, validation_generator, testing_generator):
    dropout_rate1 = trial.suggest_float('dropout_rate1', 0.2, 0.5)
    hidden_layer1 = trial.suggest_int('hidden_layer1', 512, 4096)
    hidden_layer2 = trial.suggest_int('hidden_layer2', 128, 4096)
    channel_num = trial.suggest_int('channel_num', 2, 12)
    num_feat = num_feat
    num_classes = num_classes
        # generate the model
    model = IGTD_Model(trial, hidden_layer1= hidden_layer1,
                       hidden_layer2=hidden_layer2,
                       dropout_rate1=dropout_rate1, 
                       channel_number=channel_num).to(device)
    
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
                                      training_generator= train_generator, 
                                      validation_generator = valid_generator,
                                      testing_generator = test_generator), 
                                      n_trials=100)
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
