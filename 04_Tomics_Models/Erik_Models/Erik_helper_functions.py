
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
import neptune.new as neptune

import torch.nn.functional as F


from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import precision_recall_curve,log_loss,f1_score, accuracy_score
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
import math

# ---------------------------------------------- data loading ----------------------------------------------#
def load_train_valid_data(path, train_data, valid_data, test_data = None):
    '''
    Functions loads the data frames that will be used to train classifier and assess its accuracy in predicting.
    input:
        train_data: filename of training csv file
        valid_data: filename of validation csv file
    ouput:
       L1000 training: pandas dataframe with training data
       L1000 validation: pandas dataframe with validation data
    '''
    if test_data:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation =pd.read_csv(path + valid_data, delimiter = ",")
        L1000_test = pd.read_csv(path + test_data, delimiter = ",")
        return L1000_training, L1000_validation, L1000_test
    else:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation =pd.read_csv(path + valid_data, delimiter = ",")
    return L1000_training, L1000_validation


# ---------------------------------------------- data preprocessing ----------------------------------------------#
def dict_splitting_into_tensor(df):
    '''
    Takes a dataframe and splits it into a dictionary of tensors.
    Input:
        df: pandas dataframe
    Output:
        dict: dictionary of tensors (keys: moa string name, values: tensors)
    '''
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df["moa"].unique().reshape(-1,1))          # fit the encoder to the unique values of the moa column
    one_hot_encoded = enc.transform(enc.categories_[0].reshape(-1,1)).toarray()
    dicti = {}
    for i in range(0, len(enc.categories_[0])):       # create a dictionary with the one hot encoded values
        dicti[str(enc.categories_[0][i])] = one_hot_encoded[i]
    return dicti

def normalize_func(trn, val, test = None):
    """
    Performs quantile normalization on the train, test and validation data. The QuantileTransformer
    is fitted on the train data, and transformed on test and validation data.
    
    Args:
            trn: train data - pandas dataframe.
            val: validation data - pandas dataframe.
            test: test data - pandas dataframe.
    
    Returns:
            trn_norm: normalized train data - pandas dataframe.
            val_norm: normalized validation - pandas dataframe.
            test_norm: normalized test data - pandas dataframe.
    inspired by  https://github.com/broadinstitute/lincs-profiling-complementarity/tree/master/2.MOA-prediction
    """
    norm_model = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
    #norm_model = StandardScaler()
    if test:
        trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
        val_norm = pd.DataFrame(norm_model.transform(val),index = val.index,columns = val.columns)
        test_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
        return trn_norm, val_norm, test_norm, str(norm_model)
    else:
        trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
        tst_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
        return trn_norm, tst_norm, str(norm_model)

def variance_threshold(x_train, x_val, var_threshold, x_test = None ):
    """
    This function perform feature selection on the data, i.e. removes all low-variance features below the
    given 'threshold' parameter.
    
    Args:
            x_fold_train: K-fold train data with only phenotypic/morphological features and PCs - pandas 
            dataframe.
            x_fold_val: K-fold validation data with only phenotypic/morphological features and PCs - pandas 
            dataframe.
            df_test_x_copy: test data - pandas dataframe with only phenotypic/morphological features and PCs.
    
    Returns:
            x_fold_train: K-fold train data after feature selection - pandas dataframe.
            x_fold_val: K-fold validation data after feature selection - pandas dataframe.
            df_test_x_copy: test data - pandas dataframe after feature selection - pandas dataframe.
    
    inspired by https://github.com/broadinstitute/lincs-profiling-complementarity/tree/master/2.MOA-prediction
    
    """
    var_thresh = VarianceThreshold(threshold = var_threshold) # sets a variance threshold
    var_thresh.fit(x_train) # learn empirical variances from X
    if x_test:
        x_test = x_test.loc[:,var_thresh.variances_ > var_threshold]
        x_train = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val = x_val.loc[:,var_thresh.variances_ > var_threshold]
        return x_train, x_val, x_test
    else:
        x_train = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val = x_val.loc[:,var_thresh.variances_ > var_threshold]
        return x_train, x_val

def splitting(df):
    '''Splitting data into two parts:
    1. input : the pointer showing where the transcriptomic profile is  
    2. target : labels (the correct MoA)
    
    Input:
        df: pandas dataframe with all columns.
    Output:
      input: pandas dataframe with all of the features
      target : returns the MoA class column separately, and as a string 
      '''
    
    target = df['moa']
    target = target.apply(str)
    input =  df.drop('moa', axis = 1)
    
    return input, target
def feature_selection(df_train_feat, df_val_feat, num_feat):
    Ridge_top_index_cof = np.array([676, 363, 742, 629, 590,  38, 612, 873, 448, 364, 844, 940, 914,
       400, 958, 288, 468, 827, 799, 639, 812, 298, 133,  59, 556, 586,
       398, 569, 491, 113, 709, 927, 190, 912,  35, 230, 945,  13,  58,
       168, 802, 162,  24, 826, 213, 686, 757, 124,  89, 534, 831, 353,
       235, 480,  50, 347, 471, 752, 374, 973,  10,  21, 850, 280, 658,
       574, 281, 624, 860, 202, 274, 913, 523, 762,  26, 601, 905, 191,
       362, 420, 451,   0, 129,  47, 393, 745, 439, 766, 582, 603, 506,
       446, 380, 103, 390, 733, 367, 939, 855, 772, 463, 732, 929, 149,
       641, 272, 145, 706,  41, 879, 295, 829, 160, 597,  18, 535, 898,
       832, 970,  65, 889, 627, 595, 701, 884, 901, 258, 297, 328, 293,
       332, 857, 203,  46, 350, 667,  29, 716,  83, 809, 524, 956, 383,
       730, 868, 704, 257, 659, 405,  27, 880, 792, 459, 714, 604, 690,
       148, 685, 397, 608, 114, 859,  45, 223, 560, 418, 415, 662, 101,
         8, 964, 452, 936, 728, 407, 231,  74, 504, 764, 888,  33, 773,
       689, 244, 441, 388, 406, 727, 656, 163, 540, 937, 618, 587, 327,
       354, 530, 414, 632, 867, 904, 804,  73, 170, 222,  85, 207,  22,
        96, 882, 487, 538, 580, 261, 687,  52, 536, 541, 893, 245, 562,
       503, 547, 469, 911,   2, 566,  48, 692,  81, 110, 746,  87, 607,
       754,  92, 571, 643, 915,  94, 856, 195, 321, 660, 318, 126, 592,
       819, 351,  99])
    if num_feat > 0:
        Ridge_top_index_cof = Ridge_top_index_cof[:num_feat]
        df_train_feat = df_train_feat.iloc[:,Ridge_top_index_cof]
        df_val_feat = df_val_feat.iloc[:,Ridge_top_index_cof]
   
    return df_train_feat, df_val_feat

def extract_all_cell_lines(df):
    '''
    Extract all cell lines from the dataframe
    Input:
        df: pandas dataframe with all columns.
    Output:
        dicti: dictionary with the cell lines as keys and the one hot encoded values as values
    '''
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df["cell_iname"].to_numpy().reshape(-1,1))          # fit the encoder to the unique values of the moa column
    one_hot_encoded = enc.transform(enc.categories_[0].reshape(-1,1)).toarray()
    dicti = {}
    for i in range(0, len(enc.categories_[0])):       # create a dictionary with the one hot encoded values
        dicti[str(enc.categories_[0][i])] = one_hot_encoded[i]
    return dicti


# --
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

def create_splits(train, val, test, cc_q75 = 0, cell_lines = {}):
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
    
    list_with_ans = []
    for set_split in [train, val, test]:
    # Removing transcriptomic profiles based on the correlation between different cell lines
        if cell_lines:
            profile_ids = choose_cell_lines_to_include(list(set_split.unique()), clue_sig_in_SPECS, cell_lines)
        else:
            profile_ids = clue_sig_in_SPECS[["Compound ID", "sig_id", "moa", 'cell_iname']][clue_sig_in_SPECS["Compound ID"].isin(set_split["Compound_ID"].unique())]
        list_with_ans.append(profile_ids)
    return list_with_ans[0], list_with_ans[1], list_with_ans[2]
#-------------------------------------------- visualization ----------------------------------------------# 
def val_vs_train_loss(epochs, train_loss, val_loss, now, model_name, loss_path_to_save):
    ''' 
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_loss: training loss per epoch (python list)
    val_loss: validation loss per epoch (python list)
    ''' 
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_loss, label = "train_loss")
    plt.plot(x_axis, val_loss, label = "val_loss")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.title(f'Validation versus Training Loss: {model_name}')
    plt.legend()
    # plot
    plt.savefig(loss_path_to_save + '/' + 'loss_train_val_' + model_name + now)


def val_vs_train_accuracy(epochs, train_acc, val_acc, now, model_name, acc_path_to_save):
    '''
    Plotting validation versus training loss over time
    epochs: number of epochs that the model ran (int. hyperparameter)
    train_acc: accuracy loss per epoch (python list)
    val_acc: accuracy loss per epoch (python list)
    '''
    plt.figure()
    x_axis = list(range(1, epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_acc, label = "train_acc")
    plt.plot(x_axis, val_acc, label = "val_acc")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation versus Training Accuracy: {model_name}')
    plt.legend()
    # plot
    plt.savefig(acc_path_to_save + '/' + 'acc_train_val_' + model_name + now)

def conf_matrix_and_class_report(labels_val, predictions, model_name):
    '''
    Plotting confusion matrix and classification report.
    Saves the images locally to then subsequently be uploaded to neptune.ai as a file and image.
    '''
    cf_matrix = confusion_matrix(labels_val, predictions)
    print(f' Confusion Matrix: {cf_matrix}')
    plt.figure()
    sns.heatmap(cf_matrix, annot = True, fmt='d').set(title = f'Confusion Matrix: {model_name}')
    plt.savefig("Conf_matrix.png")
   
    class_report = classification_report(labels_val, predictions)
    print(class_report)
    f = open("class_info.txt","w")
    # write file
    f.write(str(class_report) )
    # close file
    f.close()

#---------------------------------------------- model ----------------------------------------------#
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:  # if the validation loss is less than the minimum validation loss we have seen so far
            self.min_validation_loss = validation_loss  # update the minimum validation loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta): # if the validation loss is greater than the minimum validation loss we have seen so far + the minimum delta
            self.counter += 1                       # increment the counter
            if self.counter >= self.patience:       # if the counter is greater than the patience
                return True
        return False
    
# ---------------------------------------------- general  ----------------------------------------------#
def program_elapsed_time(start, end):
    '''
    Calculates the time elapsed for a program to run.
    Input:
        start: time when the program started (time.time())
        end: time when the program ended (time.time())
    Output:
        time_elapsed: time elapsed for the program to run (string). Seconds, minutes or hours depending on time elapsed'''
    program_time = round(end - start, 2) 
    print(program_time)
    if program_time > float(60) and program_time < 60*60:
        program_time =  program_time/60
        time_elapsed = str(program_time) + ' min'
    elif program_time > 60*60:
        program_time = program_time/3600
        time_elapsed = str(program_time) + ' hrs'
    else:
        time_elapsed = str(program_time) + ' sec'
    return time_elapsed