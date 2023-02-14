#!/usr/bin/env python
# coding: utf-8

# Import Statements -------------------------------------------------------------------------------------------------#
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets

from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
import pickle


# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# ----------------------------------------------------------------------------------------------------------------#

def train_test_valid_split( df, train_cpds, test_cpds, valid_cpds):
    '''
    Splitting the original data set into train and test at the compound level
    
    Input:
        df: pandas dataframe with all rows with relevant information after pre-processing/screening
        train_cpds: list of training compounds
        test_cpds: list of test compounds
        (valid_cpds): list of valid compounds
    Output:
        df_train: pandas dataframe with only rows that have training compounds
        df_test: pandas dataframe with only rows that have test compounds
        (df_valid): pandas dataframe with only rows that have valid compounds
        '''
    if valid_cpds: 
        df_train = df.loc[df["Compound_ID"].isin(train_cpds)]
        df_valid = df.loc[df["Compound_ID"].isin(valid_cpds)]
        df_test = df.loc[df["Compound_ID"].isin(test_cpds)]
        return df_train, df_test, df_valid
    
    else: 
        df_train = df.loc[df["Compound_ID"].isin(train_cpds)]
        df_test = df.loc[df["Compound_ID"].isin(test_cpds)]
        return df_train, df_test
    
def save_to_csv(df, file_name, filename_mod, compress = None):
    '''Saving train, test or valid set to specific directory with or without compression
    Input:
        df: the dataframe to be saved
        file_name: standardized naming depending on if its a training/validation/test set
        file_name_mod: the unique input by user to identify the saved csv file in the directory
        compress: compress the resulting csv file if desired.
    Output:
        CSV file in the dir_path directory
    '''
    
    dir_path = "/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/"
    
    if not os.path.exists(dir_path):
        print("Making path")
        os.mkdir(dir_path)
    df.to_csv(dir_path + file_name + '_'+ filename_mod + ".csv", index = False, compression = compress)

def save_to_pickle(dictionary, filename_mod):
    '''Saving train, test or valid set to specific directory with or without compression
    Input:
        dictionary: the dictionary to be saved
        file_name: standardized naming depending on if its a training/validation/test set
        file_name_mod: the unique input by user to identify the saved csv file in the directory
        compress: compress the resulting csv file if desired.
    Output:
        CSV file in the dir_path directory
    '''
    
    dir_path = "/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/CS_data_splits/"
    
    if not os.path.exists(dir_path):
        print("Making path")
        os.mkdir(dir_path)
    with open(dir_path + '/'+ filename_mod + ".pickle", 'wb') as handle:
        pickle.dump(dictionary, handle)


def create_splits(moas, filename_mod, perc_test, need_val = True):
    '''
    Input:
        moas: the list of moas being investigated.
        filename_mod: Name of the resulting csv file to be found.
        perc_test: The percentage of the data to be placed in the training vs test data.
        need_val: True/False: do we need a validation set?
    Output:
        2 or 3 separate csv files, saved to a separate folder. Each csv file represents training, validation or test sets-
    '''

    # read in documents
    compounds_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/compounds_v1v2.csv', delimiter = ",")
    
    # # Pre-processing Psuedo-Code
    # 1. Extract only those compounds with the correct moas.
    # 2. Prepare classes.
    # 3. Do the test, train  and validation split, making sure to shuffle
    # 4. Save the test, train and validation splits to a csv.

# -------------------------------------------- #1 --------------------------------------------------------------------------
    # Keeping only those moas that we are interested in


    data_set = compounds_v1v2[compounds_v1v2["moa"].isin(moas)]

#--------------------------------------------- #2 ------------------------------------------------------------------------
    # create dictionary where moas are associated with a number
    dictionary = {}
    for i,j in enumerate(moas):
        dictionary[j] = i
    save_to_pickle(dictionary, filename_mod + "_moa_dict")

    moa_col = data_set.columns.get_loc("moa")
    # change moa to classes using the above dictionary
    for i in range(data_set.shape[0]):
        data_set.iloc[i, moa_col] = dictionary[data_set.iloc[i, moa_col]]



    # ## Train and Test Set Splitting
    # Pseudocode
    # 1. extract all of the compounds from that have transcriptomic profiles
    # 2. split the compounds into a train, test and validation data set
    # 3. create list of compound names for each set


    compound_split = data_set


    # --------------------------- 3. splitting into training, validation and test sets----------#
    # split dataset into test and training/validation sets (10-90 split)
    compound_train_valid, compound_test, compound_train_valid_moa, test_Y = train_test_split(
    compound_split, compound_split["moa"],  stratify=compound_split["moa"], 
        shuffle = True, test_size = perc_test, random_state = 1)


    assert (int(compound_train_valid.shape[0]) + int(compound_test.shape[0])) == int(compound_split.shape[0])
    
    # if we want validation set
    if need_val:

        # Split data set into training and validation sets (1 to 9)
        # Same as above, but does split of only training data into training and validation 
        # data (in order to take advantage of stratification parameter)
        compound_train, compound_valid, moa_train, moa_valid = train_test_split(
        compound_train_valid, compound_train_valid["moa"], test_size = perc_test, shuffle= True,
            stratify = compound_train_valid["moa"],
            random_state = 62757)

        # list compounds in each set
        cmpd_trai_lst = list(compound_train["Compound_ID"])
        cmpd_vali_lst = list(compound_valid["Compound_ID"])
        cmpd_tes_lst = list(compound_test["Compound_ID"])


        assert (int(compound_train.shape[0]) + int(compound_valid.shape[0])) == int(compound_train_valid.shape[0])

        # create pandas datafame sets
        training_set, test_set, validation_set = train_test_valid_split(data_set, cmpd_trai_lst, cmpd_tes_lst, cmpd_vali_lst)

        # save to CSVS
        save_to_csv(training_set, "CS_training_set", filename_mod)
        save_to_csv(validation_set, "CS_valid_set", filename_mod)
        save_to_csv(test_set, "CS_test_set", filename_mod)
    
    # if we only want test and training set
    else:
        
        cmpd_trai_lst = list(compound_train_valid["Compound_ID"])
        cmpd_tes_lst = list(compound_test["Compound_ID"])
        
        cmpd_vali_lst = False
        training_set, test_set = train_test_valid_split(data_set, cmpd_trai_lst, cmpd_tes_lst, cmpd_vali_lst)
#--------------------------------------------- #4 ------------------------------------------------------------------------
        save_to_csv(training_set, "CS_training_set", filename_mod)
        save_to_csv(test_set, "CS_test_set", filename_mod)

if __name__ == "__main__":  
    #moas = list(input('List of moas (lst with strs) (ex: ["cyclooxygenase inhibitor", "adrenergic receptor antagonist"]): ') )
    moas = ["cyclooxygenase inhibitor", "dopamine receptor antagonist"]
    filename_mod = input('filename modifier (ex :cyclo_adr_2): ' )            
    perc_test = float(input ('Perc in test/val data (ex: 0.2): '))
    need_val = True
    create_splits(moas,filename_mod, perc_test, need_val)