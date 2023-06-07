

#!/usr/bin/env python
# coding: utf-8

# Import Statements

import numpy as np
import pandas as pd
import pickle  # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import datetime
import time
# Torch
import torch 
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import neptune.new as neptune
import os
import time
from datetime import datetime
import cv2
import os
import sys

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    choose_device,
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    adapt_training_loop, 
    adapt_test_loop,
    inputs_equalto_labels_check,
)
start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")


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

with open('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM_erik10_hq_8_12_splits.pkl', 'rb') as f:
    index_splits = pickle.load(f)
labels = pd.concat([train_labels, valid_labels, test_labels], axis=0).reset_index(drop=True)

for fold_int in range(4,5):
    print(f'Fold Iteration: {fold_int}')
    index_split_n = index_splits[fold_int]
    # split generated images into train, test and validation
    train_range = index_split_n["train"]
    valid_range = index_split_n["valid"]
    test_range = index_split_n["test"]

    # split generated labels into train, test and validation
    train_labels = labels.iloc[index_split_n["train"]].reset_index(drop=True)
    valid_labels = labels.iloc[index_split_n["valid"]].reset_index(drop=True)
    test_labels = labels.iloc[index_split_n["test"]].reset_index(drop=True)
    # checking that the number of images and labels are the same
    # Example usage:
    folder_path = '/scratch2-shared/erikep/DWTM/ImageDataset/'
    #train_range = range(dict_indexes["train"][0], dict_indexes["train"][1]+1)  # Images 1-100 for training
    #validation_range = range(dict_indexes["valid"][0],dict_indexes["valid"][1]+1)  # Images 101-125 for testing
    #test_range = range(dict_indexes["test"][0], dict_indexes["test"][1]+1)  # Images 126-150 for validation

    train_images, test_images, valid_images = load_images(folder_path, train_range, test_range, valid_range)
    inputs_equalto_labels_check(train_images, 
                                train_labels, 
                                valid_images, 
                                valid_labels, 
                                test_images, 
                                test_labels)
    using_cuda = True
    device = choose_device(using_cuda)

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
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(dict_moa))
    model = model.to(device)



    batch_size = 50
    train_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma = (0.1, 0.2))

    trainset = DWTM_profiles(train_images, train_labels["moa"], dict_moa)
    training_generator = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    validset = DWTM_profiles(valid_images, valid_labels["moa"], dict_moa)
    validation_generator = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    testset = DWTM_profiles(test_images, test_labels["moa"], dict_moa)
    test_generator = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    max_epochs = 100
    learning_rate = 0.0008271360695780358
    yn_class_weights = True
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_CL(train_labels, dict_moa, device)
    else:
        class_weights = None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,  step_size=12, gamma= 0.7949870139012913)

    # choosing loss_function 
    loss_fn_str = 'BCE'
    loss_fn_train_str = 'ols'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str,
                                                    loss_fn_train_str = loss_fn_train_str, 
                                                    class_weights=class_weights,
                                                    smoothing = 0.06895853013598055,
                                                    alpha = 0.6854878869112724 )



    # In[52]:
    # --------------------------Function to perform training, validation, testing, and assessment ------------------


    #------------------------------   Calling functions --------------------------- #
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = my_lr_scheduler,
                model_name=model_name,
                device = device,
                val_str = 'f1',
                early_patience = 50)

    #--------------------------------- Assessing model on test data ------------------------------#
    model_test = model
    model_test.load_state_dict(torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')['model_state_dict'])
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model_test,
                                            test_loader = test_generator,
                                            device = device)

    #---------------------------------------- Visual Assessment ---------------------------------# 
    val_vs_train_loss_path = val_vs_train_loss(num_epochs = num_epochs,
                                            train_loss = train_loss_per_epoch,
                                            val_loss = val_loss_per_epoch, 
                                            now = now, 
                                            model_name = model_name, 
                                            file_name = file_name, 
                                            loss_path_to_save = '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images') 
    val_vs_train_acc_path = val_vs_train_accuracy(num_epochs = num_epochs, 
                                                train_acc = train_acc_per_epoch, 
                                                val_acc = val_acc_per_epoch, 
                                                now = now,  
                                                model_name = model_name, 
                                                file_name = file_name, 
                                                acc_path_to_save ='/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Best_Tomics_Model/saved_images')


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
                        loss_fn = loss_fn,
                        all_labels = all_labels,
                        all_predictions = all_predictions,
                        dict_moa = dict_moa,
                        val_vs_train_loss_path = val_vs_train_loss_path,
                        val_vs_train_acc_path = val_vs_train_acc_path,
                        loss_fn_train = loss_fn_train)
