#!/usr/bin/env python
# coding: utf-8

# # Splits 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random

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


def main(subset, compounds_v1v2, str_help ):
    '''Does a 5 fold split of the data and saves the data to a csv file.
    Input:
        subset: string, either cyc_adr, erik10, tian10, or 0
        compounds_v1v2: pandas dataframe, contains the compounds in specs v1 and v2
        str_help: string, either _all or _clue
    Output:
        saves the data to a csv file (5 csv files)'''


    X = compounds_v1v2["Compound_ID"]
    y = compounds_v1v2["moa"]

    # subsetting the data according to usage
    if subset == "cyc_adr":
        y = y[y.isin(["cyclooxygenase inhibitor", "adrenergic receptor antagonist"])]
        X = X.iloc[y.index]
    elif subset == "erik10":
        y = y[y.isin(["cyclooxygenase inhibitor", "dopamine receptor antagonist","adrenergic receptor antagonist", "phosphodiesterase inhibitor",  "HDAC inhibitor", 
                "histamine receptor antagonist","EGFR inhibitor", "adrenergic receptor agonist", "PARP inhibitor",  "topoisomerase inhibitor"])]
        X = X.iloc[y.index]
    elif subset == "tian10":
        y = y[y.isin(['Aurora kinase inhibitor', 'tubulin polymerization inhibitor', 'JAK inhibitor', 'protein synthesis inhibitor', 'HDAC inhibitor', 
            'topoisomerase inhibitor', 'PARP inhibitor', 'ATPase inhibitor', 'retinoid receptor agonist', 'HSP inhibitor'])]
        X = X.iloc[y.index]  
    elif subset == "0":
        pass
    else:
        raise ValueError("subset must be either cyc_adr, erik10, tian10, or 0")

    skf = StratifiedKFold(n_splits = 5, shuffle = True,  random_state=5)
    skf.get_n_splits(X,y)

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
        _20_ = int(len(train_val_index)*0.8)
        train_index = random.sample(train_val_index.tolist(), _20_)
        val_index = list(set(train_val_index.tolist()) - set(train_index))
        assess_duplicates(train_index + val_index)
        train = compounds_v1v2.iloc[X.iloc[train_index].index].reset_index(drop = True)
        test = compounds_v1v2.iloc[X.iloc[test_index].index].reset_index(drop = True)
        val = compounds_v1v2.iloc[X.iloc[val_index].index].reset_index(drop = True)
        train.to_csv(file_loc + subset + str_help + '_train_' + fold_num)
        test.to_csv(file_loc + subset + str_help + '_test_' + fold_num)
        val.to_csv(file_loc + subset + str_help + '_val_' + fold_num)

    print("Done!")

if __name__ == "__main__": 

   # read in all compounds in specs v1 and v2
    all_compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")

    # read in the compounds that compounds_v1v2 have in common with clue
    clue_cmpds_in_cmpdsv1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/clue_cmpds_in_cmpdsv1v2.csv', delimiter = ",")
    all_or_clue = input("Enter all or clue: ")
    subset = input("Enter subset name (options: cyc_adr, erik10, tian10, 0): ")
    if all_or_clue == "all":
        compounds_v1v2 = all_compounds_v1v2
        str_help = "_all"
    if all_or_clue == "clue":
        compounds_v1v2 = clue_cmpds_in_cmpdsv1v2
        str_help = "_clue"
    main(subset, compounds_v1v2, str_help)