#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Functipn to split data into training, validation and test sets
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import glob   # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import os   # miscellneous operating system interfaces. This module provides a portable way of using operating system dependent functionality. If you just want to read or write a file see open(), if you want to manipulate paths, see the os.path module, and if you want to read all the lines in all the files on the command line see the fileinput module.
import random       
from tqdm import tqdm 
from tqdm.notebook import tqdm_notebook
import datetime
import time
from tabulate import tabulate

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

import torch.nn.functional as F
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve,log_loss, f1_score, accuracy_score
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN
from sklearn.feature_selection import VarianceThreshold

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
import time
import re
from torch.utils.data import WeightedRandomSampler

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (
    apply_class_weights_CL, 
    accessing_correct_fold_csv_files, 
    create_splits, 
    choose_device,
    dict_splitting_into_tensor, 
    extract_tprofile, 
    EarlyStopper, 
    val_vs_train_loss,
    val_vs_train_accuracy, 
    program_elapsed_time, 
    conf_matrix_and_class_report,
    tprofiles_gc_too_func, 
    create_terminal_table, 
    upload_to_neptune, 
    different_loss_functions, 
    Transcriptomic_Profiles_gc_too, 
    Transcriptomic_Profiles_numpy,
    set_bool_hqdose, 
    set_bool_npy, 
    FocalLoss, 
    np_array_transform,
    apply_class_weights_GE, 
    adapt_training_loop, 
    adapt_validation_loop, 
    adapt_test_loop,
    checking_veracity_of_data,
    check_overlap_sigid,
    accessing_all_folds_csv,
    pre_processing
)

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

method = 'DWTM'
file_name = "erik10_hq_8_12"
fold_index_dictionaries = []
for fold_int in range(0,5):
    print(f'Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    file_name = "erik10_hq_8_12"
    #file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    variance_thresh = 0
    normalize_c = False
    npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c, five_fold = 'True')
    df_train_features, df_val_features, df_train_labels_str, df_val_labels_str, df_test_features, df_test_labels_str, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
            clue_gene, 
            npy_exists = npy_exists,
            use_variance_threshold = variance_thresh, 
            normalize = normalize_c, 
            save_npy = save_npy,
            data_subset = file_name)
    checking_veracity_of_data(file_name, df_train_labels_str, df_val_labels_str, df_test_labels_str)
    
    fold_dict_index = {}
    for df_split, df_split_str in [(df_train_features, "train"), (df_val_features, "val"), (df_test_features, "test")]:
        if method == 'DWTM':
            df_original = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/erik10_hq_8_12_fold0')
            df_original = df_original.drop("Class", axis = 1)
        elif method == 'IGTD':
            df_original = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/IGTD_erik10_hq_8_12_fold0')
            df_original = df_original.drop("Unnamed: 0", axis = 1)
        else:
            ValueError('Method must be either DWTM or IGTD')
                # reseting index of both dataframes
        df_original = df_original.round(decimals =2)
        df_split = df_split.round(decimals = 2)
        df_original = df_original.reset_index().rename(columns={'index': 'original_index'})
        # Create a dictionary to map the current column names to their string representations
        column_mapping = {col: str(col) for col in df_split.columns}
        # Update the column names in the dataframe using the rename() function
        df_split = df_split.rename(columns=column_mapping)
        df_split.reset_index().rename(columns={'index': 'split_index'})
        # Get the common columns to merge on, excluding the 'index' column
        common_columns1 = [col for col in df_original.columns if col != 'original_index']
        common_columns2 = [col for col in df_split.columns if col != 'index']
        assert common_columns1 == common_columns2, "The columns in the original and split dataframes should match"
        # Merge the dataframes on all common columns
        merged_df = df_original.merge(df_split, on=common_columns1 )
        assert merged_df.shape[0] == df_split.shape[0], "The number of rows in the merged dataframe should match the number of rows in the split dataframe"
        # Extract the index from the merged dataframe
        matching_indices = merged_df['original_index'].values
        fold_dict_index[df_split_str] = matching_indices
    fold_index_dictionaries.append(fold_dict_index)
with open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/" + method + '_' + file_name + "_splits.pkl", 'wb') as f:
    pickle.dump(fold_index_dictionaries, f)


    