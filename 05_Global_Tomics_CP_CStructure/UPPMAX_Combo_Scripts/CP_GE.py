#!/usr/bin/env python
# coding: utf-8
import datetime
import time
import torch
import torch.nn as nn
import neptune.new as neptune
import sys
import h5py

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    adapt_training_loop, 
    adapt_test_loop,
    GE_driving_code,
    extracting_pretrained_single_models,
    CP_driving_code,
    set_parameter_requires_grad,
    extracting_Tprofiles_with_cmpdID,
    dubbelcheck_dataset_length
)
from Helper_Models import (
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           Modified_GE_Model)

from Helper_Models import (image_network)

start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_GE'
class CP_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, meta_data_CP,  checking_mechanism, dict_moa, tprofiles_df, split_sets, dict_cell_line, images_H):
        self.compound_df = compound_df
        self.meta_data_CP = meta_data_CP
        self.dict_moa = dict_moa
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
        self.images_H = images_H
        
    
    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data_CP) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data_CP)
    
    def __getitem__(self,idx):
        meta_data_idx = self.meta_data_CP.index[idx]  
        label = self.meta_data_CP["moa"][meta_data_idx]
        cmpdID = self.meta_data_CP["compound"][meta_data_idx]
        with h5py.File(self.images_H, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])
        t_profile, t_cell_line = extracting_Tprofiles_with_cmpdID(self.tprofiles_df, self.split_sets, cmpdID, self.dict_cell_line)
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return image.float(), t_profile, t_cell_line, label_tensor.float() # returns 

class CP_GE_Model(nn.Module):
    def __init__(self, modelCP, modelGE):
        super(CP_GE_Model, self).__init__()
        self.modelCP = torch.nn.Sequential(*list(modelCP.children())[:-1])
        self.modelGE = modelGE
        self.linear_layer_pre = nn.Linear(int(20 + 1280), 85)
        self.additional_layers = nn.Sequential(
            nn.Linear(in_features=85, out_features=192, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.34363036702197675, inplace=False),
            nn.Linear(in_features=192, out_features=10, bias=True)
        )
       
    def forward(self, x1in, x2in, x3in):
        x1 = self.modelCP(x1in)
        x2 = self.modelGE(x2in, x3in)
        x  = torch.cat((torch.squeeze(x1), x2), dim = 1)
        x = self.linear_layer_pre(x)
        output = self.additional_layers(x)
        return output


images_H = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
file_name = "erik10_hq_8_12"
for i in range(2, 5):
    fold_int = i
    train_np_GE, valid_np_GE, test_np_GE, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training_GE, L1000_validation_GE, L1000_test_GE = GE_driving_code(file_name, fold_int)

    training_df_CP, validation_df_CP, test_df_CP, dict_moa, cmpd_training_set, cmpd_validation_set, cmpd_test_set, images = CP_driving_code(file_name, fold_int)

#-----------------------------------------Prepping Individual Models ---------------------#
    # parameters for the dataloader
    batch_size = 10
    params = {'batch_size' : batch_size,
            'num_workers' : 3,
            'shuffle' : True,
            'prefetch_factor' : 1} 


    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    #device = torch.device('cpu')
    print(f'Training on device {device}. ' )


    # --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
    # Create datasets with relevant data and labels
    training_dataset_CP_GE = CP_GE_Dataset(compound_df = cmpd_training_set, 
                                        meta_data_CP = training_df_CP, 
                                        checking_mechanism= ["train" , "CP", fold_int],
                                        dict_moa = dict_moa, 
                                        tprofiles_df = train_np_GE, 
                                        split_sets = L1000_training_GE,
                                            dict_cell_line = dict_cell_lines, 
                                            images_H = images_H)

    valid_dataset_CP_GE = CP_GE_Dataset(compound_df =cmpd_validation_set, 
                                        meta_data_CP = validation_df_CP, 
                                        checking_mechanism = ["valid" , "CP", fold_int], 
                                        dict_moa = dict_moa, 
                                        tprofiles_df = valid_np_GE, 
                                        split_sets = L1000_validation_GE, 
                                        dict_cell_line = dict_cell_lines, 
                                        images_H = images_H)
    test_dataset_CP_GE = CP_GE_Dataset(compound_df = cmpd_test_set, 
                                    meta_data_CP = test_df_CP, 
                                    checking_mechanism = ["test" , "CP", fold_int], 
                                    dict_moa = dict_moa, 
                                    tprofiles_df = test_np_GE, 
                                    split_sets = L1000_test_GE, 
                                    dict_cell_line = dict_cell_lines, 
                                    images_H = images_H)

    # create generator that randomly takes indices from the training set
    training_generator = torch.utils.data.DataLoader(training_dataset_CP_GE, **params)
    validation_generator = torch.utils.data.DataLoader(valid_dataset_CP_GE, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset_CP_GE, **params)

    # create a model combining both models
    cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
    cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
    modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
    modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    modelCP = image_network()
    modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    modified_GE_model = Modified_GE_Model(modelGE)
    model = CP_GE_Model(modelCP, modified_GE_model)


    set_parameter_requires_grad(model, feature_extracting = True, added_layers = 1)

    learning_rate = 2.37011523414972e-05
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # loss_function
    yn_class_weights = False
    if yn_class_weights:
        class_weights = apply_class_weights_CL(training_df_CP, dict_moa, device)
    else: 
        class_weights = None
    # choosing loss_function 
    loss_fn_str = 'cross'
    loss_fn_train, loss_fn = different_loss_functions(loss_fn_str= loss_fn_str, 
                                                    class_weights=class_weights)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.4829080063660469)
    num_epochs_fe = 12
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = num_epochs_fe,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = my_lr_scheduler,
                model_name= model_name,
                device = device,
                val_str = 'f1',
                early_patience = 0)

    #----------------------------------------- Assessing model on test data -----------------------------------------#
    model_test = CP_GE_Model(modelCP, modified_GE_model)
    model_test.load_state_dict(torch.load('/home/jovyan/saved_models/' + model_name + ".pt")['model_state_dict'])
    correct, total, avg_test_loss, all_predictions, all_labels = adapt_test_loop(model = model, 
                        test_loader = test_generator, 
                        device = device)
    # ----------------------------------------- Plotting loss, accuracy, visualization of results ---------------------#

    val_vs_train_loss_path = val_vs_train_loss(num_epochs,train_loss_per_epoch, val_loss_per_epoch, now, model_name, file_name,'/home/jovyan/saved_images') 
    val_vs_train_acc_path = val_vs_train_accuracy(num_epochs, train_acc_per_epoch, val_acc_per_epoch, now,  model_name, file_name, '/home/jovyan/saved_images')


    #-------------------------------- Writing interesting info into neptune.ai----------------------------------# 
    end = time.time()

    elapsed_time = program_elapsed_time(start, end)

    create_terminal_table(elapsed_time, all_labels, all_predictions)
    upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = True,
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
                        variance_thresh = 0,
                        pixel_size = 0,
                        loss_fn_train = loss_fn_train)
