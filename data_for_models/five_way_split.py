#!/usr/bin/env python
# coding: utf-8

# # Splits 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
import re

def assess_duplicates(flat_list):
    '''
    This function takes a list and checks if there are any duplicates.
    If there are duplicates, it raises an assertion error.
    '''
    count_dict = {}
    for element in flat_list:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    num_duplicates = 0
    for count in count_dict.values():
        if count > 1:
            num_duplicates += 1
    assert num_duplicates == 0, f"Found {num_duplicates} duplicates in the list"

def check_veracity_cmpd_num(file, X):
    if file == 'tian10':
        assert X.nunique() == 121, 'Incorrect number of unique compounds'
    elif file == 'erik10': # tw0 extra
        assert X.nunique() == int(243 - 2), 'Incorrect number of unique compounds'
    elif file == 'erik10_hq': # two extra
        assert X.nunique() == 185, 'Incorrect number of unique compounds'
    elif file == 'erik10_8_12': # two extra
        assert X.nunique() == 238, 'Incorrect number of unique compounds'
    elif file == 'erik10_hq_8_12': # yes
        assert X.nunique() == 177, 'Incorrect number of unique compounds'
    elif file == 'cyc_adr':
        assert X.nunique() == 76, 'Incorrect number of unique compounds'
    elif file == 'cyc_dop':
        assert X.nunique() == 76, 'Incorrect number of unique compounds'
    else:
        raise ValueError('Please enter a valid file name')

def main(subset, file_tosplit, compounds_v1v2, str_help, hq = "False", dosage = "False", low_t = 0, high_t = 100):
    '''Does a 5 fold split of the data and saves the data to a csv file.
    Input:
        subset: string, either cyc_adr, cyc_dop, erik10, tian10, or 0
        compounds_v1v2: pandas dataframe, contains the compounds in specs v1 and v2
        str_help: string, includes clue/all, high quality profiles (hq), and/or dosage range
        hq: boolean, if True, only high quality profiles are used
        dosage: boolean, if True, only profiles within a certain dosage range are used
        low_t: float, lower bound of the dosage range
        high_t: float, upper bound of the dosage range
    Output:
        saves the data to a csv file (5 csv files)'''
    compounds_v1v2 = compounds_v1v2[compounds_v1v2.moa.str.contains("|", regex = False, na = True) == False]
    clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")
    if hq == "True" and dosage == "True":
        hq_profiles = clue_sig_in_SPECS[clue_sig_in_SPECS["is_hiq"] == 1]
        dos_hq_profiles = hq_profiles[hq_profiles.pert_dose.between(int(low_t), int(high_t))]
        compounds_for_splitting = compounds_v1v2[["Compound_ID", "moa"]][compounds_v1v2["Compound_ID"].isin(dos_hq_profiles["Compound ID"].unique())].drop_duplicates()
    elif hq == "True":
        # extract out the hq compounds that are only of high quality
        hq_profiles = clue_sig_in_SPECS[["Compound ID", "moa"]][clue_sig_in_SPECS["is_hiq"] == 1]
        compounds_for_splitting = compounds_v1v2[["Compound_ID", "moa"]][compounds_v1v2["Compound_ID"].isin(hq_profiles["Compound ID"].unique())].drop_duplicates()
    elif dosage == "True":
        dos_profiles = clue_sig_in_SPECS[["Compound ID", "moa"]][clue_sig_in_SPECS.pert_dose.between(low_t, high_t)]
        compounds_for_splitting = compounds_v1v2[["Compound_ID", "moa"]][compounds_v1v2["Compound_ID"].isin(dos_profiles["Compound ID"].unique())].drop_duplicates()
    else:
    # Remove all compounds that have multiple moas
       
    #X = compounds_v1v2["Compound_ID"]
    # y = compounds_v1v2["moa"]
    # We want to split the data based on the moa, but only for unique compounds.
        compounds_for_splitting = compounds_v1v2[["Compound_ID", "moa"]].drop_duplicates()
    # We choose which moas to use based on the subset
    if subset == "cyc_adr":
        moas = ["cyclooxygenase inhibitor", "adrenergic receptor antagonist"]
        compounds_for_splitting = compounds_for_splitting[compounds_for_splitting.moa.isin(["cyclooxygenase inhibitor", "adrenergic receptor antagonist"])]
        assert set(compounds_for_splitting.moa) == set(moas), "There might be multple moas (cyclo | adr) in the data. "
        X = compounds_for_splitting["Compound_ID"]
        y = compounds_for_splitting["moa"]
    elif subset == "cyc_dop":
        moas = ["cyclooxygenase inhibitor", "dopamine receptor antagonist"]
        compounds_for_splitting = compounds_for_splitting[compounds_for_splitting.moa.isin(["cyclooxygenase inhibitor", "dopamine receptor antagonist"])]
        assert set(compounds_for_splitting.moa) == set(moas), "There might be multple moas (cyclo | adr) in the data. "
        X = compounds_for_splitting["Compound_ID"]
        y = compounds_for_splitting["moa"]
    elif subset == "erik10":
        moas = ["cyclooxygenase inhibitor", "dopamine receptor antagonist","adrenergic receptor antagonist", "phosphodiesterase inhibitor",  "HDAC inhibitor", 
                "histamine receptor antagonist","EGFR inhibitor", "adrenergic receptor agonist", "PARP inhibitor",  "topoisomerase inhibitor"]
        compounds_for_splitting = compounds_for_splitting[compounds_for_splitting.moa.isin(["cyclooxygenase inhibitor", "dopamine receptor antagonist","adrenergic receptor antagonist",
                 "phosphodiesterase inhibitor",  "HDAC inhibitor", "histamine receptor antagonist","EGFR inhibitor", "adrenergic receptor agonist", "PARP inhibitor",  
                 "topoisomerase inhibitor"])]
        assert set(compounds_for_splitting.moa) == set(moas), "There might be multple moas (cyclo | adr) in the data. "
        X = compounds_for_splitting["Compound_ID"]
        y = compounds_for_splitting["moa"]
    elif subset == "tian10":
        moas = ['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor',
            'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor']
        compounds_for_splitting = compounds_for_splitting[compounds_for_splitting.moa.isin(['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 
            'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor'])]
        assert set(compounds_for_splitting.moa) == set(moas), "There might be multple moas (cyclo | adr) in the data. "
        X = compounds_for_splitting["Compound_ID"]
        y = compounds_for_splitting["moa"]
    elif subset == "0":
        moas = compounds_for_splitting.moa.unique()
        
        X = compounds_for_splitting["Compound_ID"]
        y = compounds_for_splitting["moa"]
    else:
        raise ValueError("subset must be either cyc_adr, erik10, tian10, or 0")
   
    #check_veracity_cmpd_num(file_tosplit, X)

    skf = StratifiedKFold(n_splits = 5, shuffle = True,  random_state=5)
    y = y.astype(object).replace(np.nan, 'None')


    # assessing that splitting has occurred correctly
    trainval_indices = []
    test_indices = []
    for i, (train_index, test_index) in enumerate(skf.split(X,y)):
        # make sure there is no data leakeage between train and test
        assess_duplicates(train_index.tolist() + test_index.tolist())
        trainval_indices = trainval_indices + train_index.tolist()
        test_indices = test_indices + test_index.tolist()
    assess_duplicates(test_indices)
    # make sure that all indices have been in the test set at least one time
    assert len(test_indices) == len(y) 

    #compounds_v1v2.iloc[test_index].reset_index(drop = True)

    for i, (train_val_index, test_index) in enumerate(skf.split(X,y)):
        file_loc = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/' + subset + '/'
        fold_num = 'fold_' + str(i) + '.csv'
        _70_ = int(len(train_val_index)*0.70)
        train_index = random.sample(train_val_index.tolist(), _70_)
        val_index = list(set(train_val_index.tolist()) - set(train_index))
        assess_duplicates(train_index + val_index)
        # 1. use split indices to extract out the compound ids for each set
        # 2. do is in on the compound ids to get the indices of the rows in compounds_v1v2 to be included in each set
        # 3. use those indices to get the original row data from compounds_v1v2
    
        train = compounds_v1v2[compounds_v1v2["Compound_ID"].isin(list(X.iloc[train_index]))].reset_index(drop = True)
        test = compounds_v1v2[compounds_v1v2["Compound_ID"].isin(list(X.iloc[test_index]))].reset_index(drop = True)
        val = compounds_v1v2[compounds_v1v2["Compound_ID"].isin(list(X.iloc[val_index]))].reset_index(drop = True)
        
        # removes moa's that are not in the subset
        train = train[train.moa.isin(moas)]
        test = test[test.moa.isin(moas)]
        val = val[val.moa.isin(moas)]

        # checking that compounds do not overlap between sets
        inter1 = set(list(train["Compound_ID"])) & set(list(val["Compound_ID"]))
        inter2 = set(list(train["Compound_ID"])) & set(list(test["Compound_ID"]))
        inter3 = set(list(val["Compound_ID"])) & set(list(test["Compound_ID"]))
        assert len(inter1) + len(inter2) + len(inter3) == 0, ("There is an intersection between the training, validation and test sets")

        # checking that the moa's are the same in each set
        assert set(train.moa.unique()) == set(val.moa.unique()) == set(test.moa.unique()) == set(moas), ("The moa's are not the same in each set")
        train.to_csv(file_loc + subset + str_help + '_train_' + fold_num)
        test.to_csv(file_loc + subset + str_help + '_test_' + fold_num)
        val.to_csv(file_loc + subset + str_help + '_val_' + fold_num)
    
    print("Done!")


if __name__ == "__main__": 

   # read in all compounds in spallecs v1 and v2
    all_compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")

    # read in the compounds that compounds_v1v2 have in common with clue
    clue_cmpds_in_cmpdsv1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/clue_cmpds_in_cmpdsv1v2.csv', delimiter = ",")
    all_or_clue = input("Enter all or clue: ")
    file_tosplit = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
    subset = file_tosplit 
    hq, dose = 'False', 'False'
    if re.search('hq', subset):
        hq = 'True'
        subset = re.sub('_hq', '', subset)
    if re.search('8', subset):
        dose = 'True'
        subset = re.sub('_8_12', '', subset)
    if dose == "True":
        low_t = 8
        high_t = 12
        dos = "_" + str(low_t) + "_" + str(high_t) + "_"
    else:
        dos = "" 
        low_t = 0
        high_t = 100  
    if hq == "True":
        add = "_hq"
    else:
        add = ""
    if all_or_clue == "all":
        compounds_v1v2 = all_compounds_v1v2
        str_help = "_all" + add + dos
    elif all_or_clue == "clue":
        compounds_v1v2 = clue_cmpds_in_cmpdsv1v2
        str_help = "_clue" + add + dos
    else:
        raise ValueError("Please enter all or clue")
    main(subset, file_tosplit, compounds_v1v2, str_help, hq, dose, low_t, high_t)