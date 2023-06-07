

#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pickle
import datetime
import time

# Torch
import torch
from torchvision import transforms
import torch.nn as nn
# Neptune
import neptune.new as neptune

import time
import datetime
import numpy as np

from datetime import datetime
import sys
import optuna
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    choose_device,
    different_loss_functions,  
    adapt_training_loop,  
    inputs_equalto_labels_check,
)


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


last_training_index = train_labels.shape[0]
last_validat_index = last_training_index + valid_labels.shape[0]
last_test_index = generated_images.shape[0]
# split generated images into train, test and validation
train_images = generated_images[:last_training_index]
valid_images = generated_images[last_training_index: last_validat_index]
test_images = generated_images[last_validat_index: last_test_index]


# checking that the number of images and labels are the same
inputs_equalto_labels_check(train_images, 
                            train_labels, 
                            valid_images, 
                            valid_labels, 
                            test_images, 
                            test_labels)
assert samples[last_training_index] == '0'
assert samples[last_validat_index] == '0'
#assert samples[last_test_index -1 ] ==  '2706'

using_cuda = True
device = choose_device(using_cuda)


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
        self.conv2 = nn.Conv2d(in_channels=channel_number, out_channels=16, kernel_size=(2,2), padding=1)
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

batch_size = 50
train_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma = (0.1, 0.2))

trainset = IGTD_profiles(train_images, train_labels["moa"], dict_moa)
train_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

validset = IGTD_profiles(valid_images, valid_labels["moa"], dict_moa)
valid_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

testset = IGTD_profiles(test_images, test_labels["moa"], dict_moa)
test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# --------------------------Function to perform training, validation, testing, and assessment ------------------


def objectiv(trial, num_feat, num_classes, training_generator, validation_generator, testing_generator):
    dropout_rate1 = trial.suggest_float('dropout_rate1', 0.2, 0.5)
    hidden_layer1 = trial.suggest_int('hidden_layer1', 512, 4096)
    hidden_layer2 = trial.suggest_int('hidden_layer2', 128, 4096)
    channel_num = trial.suggest_int('channel_num', 2, 12)
    num_feat = num_feat
    num_classes = num_classes
        # generate the model
    model = IGTD_Model( channel_number=channel_num, hidden_layer1= hidden_layer1,
                       hidden_layer2=hidden_layer2,
                       dropout_rate1=dropout_rate1).to(device)
    
    # generate the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))


    yn_class_weights = trial.suggest_categorical('yn_class_weights', [True, False])
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_CL(train_labels, dict_moa, device)
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'focal', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false','ols'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        

    max_epochs = 350

    
#------------------------------   Calling functions --------------------------- #
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = scheduler,
                model_name=model_name,
                device = device,
                val_str = 'f1',
                early_patience = 20)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return max(val_f1_score_per_epoch)

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objectiv(trial, num_feat = 978, 
                                      num_classes = 10, 
                                      training_generator= train_generator, 
                                      validation_generator = valid_generator,
                                      testing_generator = test_generator), 
                                      n_trials=150)
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
