#!/usr/bin/env python
# coding: utf-8

import time
from sklearn.metrics import f1_score 
import datetime
import torch
import torch.nn as nn
import neptune.new as neptune
import sys

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    choose_device,
    smiles_to_array,
    GE_driving_code,
    extracting_pretrained_single_models,
    returning_smile_string,
    dubbelcheck_dataset_length,
    ensemble_predict2
  
)

from Helper_Models import (
                           Cell_Line_Model,
                           Tomics_and_Cell_Line_Model,
                           CNN_Model,
                           Chemical_Structure_Model,)

class CS_GE_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, tprofiles_df, split_sets, dict_moa, dict_cell_line, checking_mechanism):
        self.compound_df = compound_df
        self.tprofiles_df = tprofiles_df
        self.split_sets = split_sets
        self.dict_moa = dict_moa
        self.dict_cell_line = dict_cell_line
        self.check = checking_mechanism
    def __len__(self):
        check_criteria = self.check
        assert len(self.tprofiles_df) == dubbelcheck_dataset_length(*check_criteria)
        return len(self.tprofiles_df)
    
    def __getitem__(self,idx):
        '''Retrieving the compound '''
        t_profile = self.tprofiles_df.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.tprofiles_df.iloc[idx, -1]
        CID = self.split_sets["Compound ID"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        cell_line_key = self.split_sets["cell_iname"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        CID = CID.iloc[0]
        cell_line_key = cell_line_key.iloc[0]
        t_cell_line = torch.tensor(self.dict_cell_line[cell_line_key])
        smile_string = returning_smile_string(self.compound_df,CID)
        compound_array = smiles_to_array(smile_string)
        if compound_array.shape[0] != 2048:
            raise ValueError("Compound array is not the correct size")
        assert not torch.isnan(compound_array).any(), "NaN value found in compound array"
        label_tensor = torch.from_numpy(self.dict_moa[moa_key])                  # convert label to number
        t_profile_features = torch.tensor(t_profile) 
        return compound_array.float(), t_profile_features, t_cell_line.float(), label_tensor.float() # returns 
        
class CS_GE_Model_Soft_Voting(nn.Module):
    def __init__(self, modelCS, modelGE,):
        super(CS_GE_Model_Soft_Voting, self).__init__()
        self.modelCS = modelCS
        self.modelGE = modelGE

    def forward(self, x1in, x2in, x3in):
        if x1in.shape[1] != 2048:
            raise ValueError("The input shape for the compound is not correct")
        x1 = self.modelGE(x2in, x3in)
        x2 = self.modelCS(x1in)
        return x1, x2


#----------------------------------------- pre-processing -----------------------------------------#
start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
model_name = 'CS_GE'
print("Begin Training")

#---------------------------------------------------------------------------------------------------------------------------------------#

now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")

batch_size = 50
WEIGHT_DECAY = 1e-5
learning_rate = 0.0000195589 
# parameters
params = {'batch_size' : batch_size,
         'num_workers' : 3,
         'prefetch_factor' : 2,
         'shuffle' : True} 
          


file_name = 'erik10_hq_8_12'
f1_scores_folds = []
for i in range(5):
    fold_int = i
    train_np, valid_np, test_np, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set_cmpds, validation_set_cmpds, test_set_cmpds, L1000_training, L1000_validation, L1000_test = GE_driving_code(file_name, fold_int)

    using_cuda = True 
    device = choose_device(using_cuda)         

    # -----------------------------------------Prepping Ensemble Model ---------------------#

    # Create datasets with relevant data and labels
    training_dataset_CSGE = CS_GE_Dataset(training_set_cmpds, train_np, L1000_training, dict_moa, dict_cell_lines, ["train" , "GE", fold_int])
    valid_dataset_CSGE = CS_GE_Dataset(validation_set_cmpds, valid_np, L1000_validation, dict_moa, dict_cell_lines, ["valid" , "GE", fold_int])
    test_dataset_CSGE = CS_GE_Dataset(test_set_cmpds, test_np, L1000_test, dict_moa, dict_cell_lines, ["test" , "GE", fold_int])

    # create generator that randomly takes indices from the training set
    training_generator = torch.utils.data.DataLoader(training_dataset_CSGE, **params)
    validation_generator = torch.utils.data.DataLoader(valid_dataset_CSGE, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset_CSGE, **params)

    cell_line_model = Cell_Line_Model(num_features = num_cell_lines, num_targets = 5).to(device)
    cnn_model = CNN_Model(num_features = 978, num_targets = 15, hidden_size= 4096).to(device)
    modelGE = Tomics_and_Cell_Line_Model(cnn_model, cell_line_model, num_targets = 10)
    modelGE.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    modelCS = Chemical_Structure_Model(num_features = 2048, num_targets = 10)
    modelCS.load_state_dict(torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = fold_int), map_location=torch.device('cpu'))['model_state_dict'])
    state_1_CS = torch.load(extracting_pretrained_single_models(single_model_str = "CS", fold_num = fold_int), map_location=torch.device('cpu'))
    state_2_GE = torch.load(extracting_pretrained_single_models(single_model_str = "GE", fold_num = fold_int), map_location=torch.device('cpu'))
    state_1_f1 = state_1_CS['f1_score']
    state_2_f1 = state_2_GE['f1_score']
    total_states_f1 = state_1_f1 + state_2_f1
    model = CS_GE_Model_Soft_Voting(modelCS, modelGE)
    # CP then CS
    predictions, labels = ensemble_predict2(
        test_generator=test_generator,
        model=model,
        weights = [state_2_f1/total_states_f1, state_1_f1/total_states_f1],
        device=device)

    f1_scores_folds.append(f1_score(predictions,labels, average='macro'))
print(f1_scores_folds)
