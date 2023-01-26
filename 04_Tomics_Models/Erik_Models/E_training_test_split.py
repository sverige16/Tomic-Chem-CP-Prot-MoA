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


# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torchinfo

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
        df_train = df.loc[df["Compound ID"].isin(train_cpds)]
        df_valid = df.loc[df["Compound ID"].isin(valid_cpds)]
        df_test = df.loc[df["Compound ID"].isin(test_cpds)]
        return df_train, df_test, df_valid
    
    else: 
        df_train = df.loc[df["Compound ID"].isin(train_cpds)]
        df_test = df.loc[df["Compound ID"].isin(test_cpds)]
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
    
    dir_path = "/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/data_split_csvs/"
    
    if not os.path.exists(dir_path):
        print("Making path")
        os.mkdir(dir_path)
    df.to_csv(dir_path + file_name + '_'+ filename_mod + ".csv", index = False, compression = compress)

def choose_cell_lines_to_include(moas, clue_sig_in_SPECS, MoAs_2_correlated):
    '''
    Returns a pandas dataframe which includes only the information of those entries that have the correct cell line and moa.

    Input:
        moas: the list of moas being investigated
        clue_sig_in_SPECS: the pandas dataframe with information on the small molecules found in SPECSv1/v2 and clue.io
        MoAs_2_correlated: a dictionary, where the key is the name of the moa and value is a list with the names of cell lines to be included.
    Output:
        pandas dataframe with 4 columns representing transcriptomic profiles with the correct cell line and moa.
    '''
    together = []
    for i in moas:
        bro = MoAs_2_correlated[i]
        svt = clue_sig_in_SPECS[clue_sig_in_SPECS["moa"]== i]
        yep = svt[svt["cell_iname"].isin(bro)]
        together.append(yep)
    allbo = pd.concat(together)
    allbo = allbo[["Compound ID", "sig_id", "moa", "cell_iname"]]
    return allbo

def create_splits(moas, filename_mod, perc_test, cc_q75, need_val = True, cell_lines = {}):
    '''
    Input:
        moas: the list of moas being investigated.
        filename_mod: Name of the resulting csv file to be found.
        perc_test: The percentage of the data to be placed in the training vs test data.
        cc_q75: Threshold for 75th quantile of pairwise spearman correlation for individual, level 4 profiles.
        need_val: True/False: do we need a validation set?
        cell_lines: a dictionary, where the key is the name of the moa and value is a list with the names of cell lines to be included.
            Default is empty. Ex. "{"cyclooxygenase inhibitor": ["A375", "HA1E"], "adrenergic receptor antagonist" : ["A375", "HA1E"] }"
    Output:
        2 or 3 separate csv files, saved to a separate folder. Each csv file represents training, validation or test sets-
    '''

    # read in documents
    clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")
    
    # # Pre-processing Psuedo-Code
    # 1. Do pre-processing to extract relevant transcriptomic profiles with MoAs of interest from the GCTX document
    # 2. Prepare classes.
    # 3. Do the test, train  and validation split, making sure to shuffle
    # 4. Save the test, train and validation splits to a csv.

# -------------------------------------------- #1 --------------------------------------------------------------------------
    # Removing transcriptomic profiles based on the correlation of the level 4 profiles
    if cc_q75 > 0:
        clue_sig_in_SPECS = clue_sig_in_SPECS[clue_sig_in_SPECS["cc_q75"] > cc_q75]
    
    # Removing transcriptomic profiles based on the correlation between different cell lines
    if cell_lines:
        profile_ids = choose_cell_lines_to_include(moas, clue_sig_in_SPECS, cell_lines)
    else:
        profile_ids = clue_sig_in_SPECS[["Compound ID", "sig_id", "moa", 'cell_iname']][clue_sig_in_SPECS["moa"].isin(moas)]

#--------------------------------------------- #2 ------------------------------------------------------------------------
    # create dictionary where moas are associated with a number
    dictionary = {}
    for i,j in enumerate(moas):
        dictionary[j] = i

    # change moa to classes using the above dictionary
    for i in range(profile_ids.shape[0]):
        profile_ids.iloc[i, 2] = dictionary[profile_ids.iloc[i, 2]]



    # ## Train and Test Set Splitting
    # Pseudocode
    # 1. extract all of the compounds from that have transcriptomic profiles
    # (could probably add check to see how many transcriptomic profiles a compound has)
    # 2. split the compounds into a train, test and validation data set
    # 3. create list of compound names for each set


    compound_split = profile_ids.drop_duplicates(subset=["Compound ID"])


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
        cmpd_trai_lst = list(compound_train["Compound ID"])
        cmpd_vali_lst = list(compound_valid["Compound ID"])
        cmpd_tes_lst = list(compound_test["Compound ID"])


        assert (int(compound_train.shape[0]) + int(compound_valid.shape[0])) == int(compound_train_valid.shape[0])

        # create pandas datafame sets
        training_set, test_set, validation_set = train_test_valid_split(profile_ids, cmpd_trai_lst, cmpd_tes_lst, cmpd_vali_lst)

        # save to CSVS
        save_to_csv(training_set, "L1000_training_set", filename_mod)
        save_to_csv(validation_set, "L1000_valid_set", filename_mod)
        save_to_csv(test_set, "L1000_test_set", filename_mod)
    
    # if we only want test and training set
    else:
        
        cmpd_trai_lst = list(compound_train_valid["Compound ID"])
        cmpd_tes_lst = list(compound_test["Compound ID"])
        
        cmpd_vali_lst = False
        training_set, test_set = train_test_valid_split(profile_ids, cmpd_trai_lst, cmpd_tes_lst, cmpd_vali_lst)
#--------------------------------------------- #4 ------------------------------------------------------------------------
        save_to_csv(training_set, "L1000_training_set", filename_mod)
        save_to_csv(test_set, "L1000_test_set", filename_mod)

if __name__ == "__main__":  
    #moas = list(input('List of moas (lst with strs) (ex: ["cyclooxygenase inhibitor", "adrenergic receptor antagonist"]): ') )
    moas = ["cyclooxygenase inhibitor", "dopamine receptor antagonist"]
    filename_mod = input('filename modifier (ex :cyclo_adr_2): ' )            
    perc_test = float(input ('Perc in test/val data (ex: 0.2): '))
    need_val = False
    cc_q75 = float(input('Threshold for 75th quantile of pairwise spearman correlation (ex: 0.2; zero -> no threshold): '))
    cell_lines = {"cyclooxygenase inhibitor": ["PC3"], 
 "dopamine receptor antagonist" : ["PC3"] }

    create_splits(moas,filename_mod, perc_test, cc_q75, need_val, cell_lines)