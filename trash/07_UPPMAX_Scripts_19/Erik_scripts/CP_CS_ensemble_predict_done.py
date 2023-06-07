
# import statements
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score    
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
    smiles_to_array,
    extracting_pretrained_single_models,
    CP_driving_code,
    returning_smile_string,
    dubbelcheck_dataset_length,
    ensemble_predict2
)
from Helper_Models import (image_network, Chemical_Structure_Model)



#!/usr/bin/env python
# coding: utf-8

# !pip install rdkit-pypi


class CS_CP_Model_Soft_Vote(nn.Module):
    def __init__(self, modelCP, modelCS):
        super(CS_CP_Model_Soft_Vote, self).__init__()
        self.modelCP = modelCP
        self.modelCS = modelCS
        
    def forward(self, x1, x2):
        x1 = self.modelCP(x1)
        x2 = self.modelCS(x2)
    
        return x1, x2
    
class CS_CP_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 meta_data,  
                 checking_mechanism, 
                 dict_moa, 
                compound_df,
                images_np
                 ):
        self.meta_data = meta_data
        self.dict_moa = dict_moa
        self.check = checking_mechanism
        self.compound_df = compound_df
        self.hdf5_file = images_np

    def __len__(self):
        check_criteria = self.check
        assert len(self.meta_data) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.meta_data)
    
    def __getitem__(self,idx):
        '''Retrieving the compound'''
        meta_data_idx = self.meta_data.index[idx]  
        label = self.meta_data["moa"][meta_data_idx]
        cmpdID = self.meta_data["compound"][meta_data_idx]
        
        with h5py.File(self.hdf5_file, 'r') as f:
            image = torch.from_numpy(f['CP_data'][meta_data_idx])

        smile_string = returning_smile_string(self.compound_df, cmpdID)
        compound_array = smiles_to_array(smile_string) 
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        return torch.squeeze(image.float()), compound_array.float(), label_tensor.float() # returns 


#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = 'CP_CS'
#---------------------------------------------------------------------------------------------------------------------------------------#---------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

file_name = 'erik10_hq_8_12'
f1_scores_folds = []
for i in range(5):
    fold_int = i
    training_meta_data, validation_meta_data, test_df_meta_data, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, images = CP_driving_code(file_name, fold_int)
    images = '/home/jovyan/scratch-shared/erikp/CP_file.h5'
    # --------------------------------- Prepping Dataloaders and Datasets -------------------------------#
    # Create datasets with relevant data and labels
    training_dataset_CSCP = CS_CP_Dataset(meta_data =training_meta_data,   
                    checking_mechanism = ["train" , "CP", fold_int], 
                    dict_moa = dict_moa, 
                    compound_df = training_set_cmpds,
                    images_np = images)
    valid_dataset_CSCP = CS_CP_Dataset(
                    meta_data = validation_meta_data, 
                    checking_mechanism = ["valid" , "CP", fold_int],
                    dict_moa = dict_moa, 
                    compound_df = validation_set_cmpds,
                    images_np = images)
    test_dataset_CSCP = CS_CP_Dataset(
                    meta_data = test_df_meta_data,  
                    checking_mechanism = ["test" , "CP", fold_int],
                    dict_moa = dict_moa, 
                    compound_df = test_set_cmpds,
                    images_np = images)

    # parameters for the dataloader
    batch_size = 7
    params = {'batch_size' : batch_size,
            'num_workers' : 2,
            'shuffle' : True,
            'prefetch_factor' : 1} 


    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    #device = torch.device('cpu')
    print(f'Training on device {device}. ' )

    # create generator that randomly takes indices from the training set
    training_generator = torch.utils.data.DataLoader(training_dataset_CSCP, **params)
    validation_generator = torch.utils.data.DataLoader(valid_dataset_CSCP, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset_CSCP, **params)

    # load individual models
    print("Loading Pretrained Models...")
    modelCS = Chemical_Structure_Model(num_features = 2048, num_targets=  10)
    modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    modelCP = image_network()
    modelCP.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])

    state_1_CS = torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = fold_int), map_location=torch.device('cpu'))
    state_2_CP = torch.load(extracting_pretrained_single_models(single_model_str = "CP", fold_num = fold_int), map_location=torch.device('cpu'))
    state_1_f1 = state_1_CS['f1_score']
    state_2_f1 = state_2_CP['f1_score']
    total_states_f1 = state_1_f1 + state_2_f1
    model = CS_CP_Model_Soft_Vote(modelCP, modelCS)
    # CP then CS
    predictions, labels = ensemble_predict2(
        test_generator=test_generator,
        model=model,
        weights = [state_2_f1/total_states_f1, state_1_f1/total_states_f1],
        device=device)

    f1_scores_folds.append(f1_score(predictions,labels, average='macro'))
print(f1_scores_folds)
