
#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import f1_score
import datetime
import time
# Torch
import torch
import torch.nn as nn
import neptune.new as neptune
import sys
import h5py
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    GE_driving_code,
    extracting_pretrained_single_models,
    CP_driving_code,
    ensemble_predict2,
    dubbelcheck_dataset_length,
    extracting_Tprofiles_with_cmpdID
)
from Helper_Models import (
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                        )

from Helper_Models import (image_network)
#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_GE'
#---------------------------------------------------------------------------------------------------------------------------------------#
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
        '''Retrieving the compound '''
        meta_data_idx = self.meta_data_CP.index[idx]  
        label = self.meta_data_CP["moa"][meta_data_idx]
        cmpdID = self.meta_data_CP["compound"][meta_data_idx]
        with h5py.File(self.images_H, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])
        t_profile, t_cell_line = extracting_Tprofiles_with_cmpdID(self.tprofiles_df, self.split_sets, cmpdID, self.dict_cell_line)
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return image.float(), t_profile, t_cell_line, label_tensor.float() # returns 
    
class CP_GE_Model_Soft_Vote(nn.Module):
    def __init__(self, modelCP, modelGE):
        super(CP_GE_Model_Soft_Vote, self).__init__()
        self.modelCP = modelCP
        self.modelGE = modelGE
    
       
    def forward(self, x1in, x2in, x3in):
        x1 = self.modelCP(x1in)
        x2 = self.modelGE(x2in, x3in)
        return x1, x2
    
images_H = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
file_name = 'erik10_hq_8_12'
f1_scores_folds = []
for i in range(5):
    fold_int = i
    file_name = "erik10_hq_8_12"
    train_np_GE, valid_np_GE, test_np_GE, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training_GE, L1000_validation_GE, L1000_test_GE = GE_driving_code(file_name, fold_int)

    training_df_CP, validation_df_CP, test_df_CP, dict_moa, cmpd_training_set, cmpd_validation_set, cmpd_test_set, images = CP_driving_code(file_name, fold_int)

    # download csvs with all the data pre split

    # -----------------------------------------Prepping Individual Models ---------------------#
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
     # load individual models
    print("Loading Pretrained Models...")
    cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
    cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
    modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
    modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    modelCP = image_network()
    modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    state_1_GE = torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = fold_int), map_location=torch.device('cpu'))
    state_2_CP = torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = fold_int), map_location=torch.device('cpu'))
    state_1_f1 = state_1_GE['f1_score']
    state_2_f1 = state_2_CP['f1_score']
    total_states_f1 = state_1_f1 + state_2_f1
    model = CP_GE_Model_Soft_Vote(modelCP, modelGE)
    print(state_1_f1, state_2_f1)
    # CP then CS
    predictions, labels = ensemble_predict2(
        test_generator=test_generator,
        model=model,
        weights = [state_2_f1/total_states_f1, state_1_f1/total_states_f1],
        device=device)

    f1_scores_folds.append(f1_score(predictions,labels, average='macro'))
print(f1_scores_folds)
