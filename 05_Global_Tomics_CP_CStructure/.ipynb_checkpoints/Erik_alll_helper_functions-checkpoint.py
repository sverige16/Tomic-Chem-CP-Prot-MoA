from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem

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
import cv2 

# ---------------------------------------------- data loading ----------------------------------------------#
# ----------------------------------------------------------------------------------------------------------#
def load_train_valid_data(path, train_data, valid_data, test_data = None):
    '''
    Functions loads the data frames that will be used to train classifier and assess its accuracy in predicting.
    input:
        train_data: filename of training csv file
        valid_data: filename of validation csv file
        (optional) test_data: filename of test csv file (if test_data is not None
        path: path to the folder where the data is stored
    ouput:
       L1000 training: pandas dataframe with training data
       L1000 validation: pandas dataframe with validation data
       (optional) L1000 test: pandas dataframe with test data (if test_data is not None)
    '''
    if test_data:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation = pd.read_csv(path + valid_data, delimiter = ",")
        L1000_test = pd.read_csv(path + test_data, delimiter = ",")
        return L1000_training, L1000_validation, L1000_test
    else:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation = pd.read_csv(path + valid_data, delimiter = ",")
    return L1000_training, L1000_validation


# ---------------------------------------------- data preprocessing ----------------------------------------------#

def normalize_data(trn, val, test = pd.DataFrame()):
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
    if test.shape[0] == 0:
        trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
        tst_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
        return trn_norm, tst_norm, str(norm_model) 
    else:
        trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
        val_norm = pd.DataFrame(norm_model.transform(val),index = val.index,columns = val.columns)
        test_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
        return trn_norm, val_norm, test_norm, str(norm_model)

def variance_threshold(x_train, x_val, var_threshold, x_test = pd.DataFrame() ):
    """
    This function perform feature selection on the data, i.e. removes all low-variance features below the
    given 'threshold' parameter.
    
    Input:
        x_train: training data - pandas dataframe.
        x_val: validation data - pandas dataframe.
        var_threshold: variance threshold - float. (e.g. 0.8). All features with variance below this threshold will be removed.
        (optional) x_test: test data - pandas dataframe.
    
    Output:
        x_train: training data - pandas dataframe.
        x_val: validation data - pandas dataframe.
        (optional) x_test: test data - pandas dataframe.

    inspired by https://github.com/broadinstitute/lincs-profiling-complementarity/tree/master/2.MOA-prediction
    
    """
    var_thresh = VarianceThreshold(threshold = var_threshold) # sets a variance threshold
    var_thresh.fit(x_train) # learn empirical variances from X
    if x_test.shape[0] == 0:
        x_train = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val = x_val.loc[:,var_thresh.variances_ > var_threshold]
        return x_train, x_val
    else:
        x_test = x_test.loc[:,var_thresh.variances_ > var_threshold]
        x_train = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val = x_val.loc[:,var_thresh.variances_ > var_threshold]
        return x_train, x_val, x_test
    

# ----------------------------------------- Visualization of Model Results ----------------------------------------------# 
# -----------------------------------------------------------------------------------------------------------------------#
def val_vs_train_loss(epochs, train_loss, val_loss, now, model_name, loss_path_to_save):
    ''' 
    Plotting validation versus training loss over time
    Input:
        epochs: number of epochs that the model ran (int. hyperparameter)
        train_loss: training loss per epoch (python list)
        val_loss: validation loss per epoch (python list)
        now: current time (string)
        model_name: name of the model (string)
        loss_path_to_save: path to save the plot (string)
    Output:
        Plot of validation versus training loss over time (matplotlib plot, png)
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
    return loss_path_to_save + '/' + 'loss_train_val_' + model_name + now

def val_vs_train_accuracy(epochs, train_acc, val_acc, now, model_name, acc_path_to_save):
    '''
    Plotting validation versus training accuracy over time
    Input:
        epochs: number of epochs that the model ran (int. hyperparameter)
        train_loss: training loss per epoch (python list)
        val_loss: validation loss per epoch (python list)
        now: current time (string)
        model_name: name of the model (string)
        acc_path_to_save: path to save the plot (string)
    Output:
        Plot of validation versus training accuracy over time (matplotlib plot, png)
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
    return acc_path_to_save + '/' + 'acc_train_val_' + model_name + now

def conf_matrix_and_class_report(labels_val, predictions, model_name, dict_moa = None):
    '''
    Creating a confusion matrix and classification report.
    Input:
        labels_val: validation labels (list)
        predictions: predictions of the model (list)
        model_name: name of the model (string)
    Output:
        confusion matrix (matplotlib plot, png)
        classification report (txt file)
        Saves the image/file locally, which is thensubsequently be uploaded to neptune.ai as a file and image.
    '''
    list_of_MoA_names = [0] * len(dict_moa)
    for key, value in dict_moa.items():
        list_of_MoA_names[np.argmax(value)] = key[0:4]
    cf_matrix = confusion_matrix(labels_val, predictions)
    print(f' Confusion Matrix: {cf_matrix}')
    ax = plt.figure().gca()
    sns.heatmap(cf_matrix, annot = True, fmt='d', ax = ax).set(title = f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
    ax.set_xticklabels(list_of_MoA_names); ax.set_yticklabels(list_of_MoA_names)
    plt.savefig("Conf_matrix")
   
    class_report = classification_report(labels_val, predictions, target_names= list_of_MoA_names)
    print(class_report)
    f = open("class_info.txt","w")
    # write file
    f.write(str(class_report) )
    # close file
    f.close()

#---------------------------------------------- Early Stopper for Model  ----------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
class EarlyStopper:
    '''
    Early stopping class to stop training when the validation loss is not decreasing.
    '''
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
    
# ---------------------------------------------- General  ----------------------------------------------#
#------------------------------------------------------------------------------------------------------#
def program_elapsed_time(start, end):
    '''
    Calculates the time elapsed for a program to run.
    Input:
        start: time when the program started (time.time())
        end: time when the program ended (time.time())
    Output:
        time_elapsed: time elapsed for the program to run (string). Seconds, minutes or hours depending on time elapsed
    '''
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

def apply_class_weights(training_set, device):
    """
    Applies class weights to the training set
    Input:
        training_set: training set (pandas dataframe)
        device: device to run the model on (string)
    Output:
        class_weights: class weights (torch tensor)
    """
    counts = training_set.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    class_weights = []   # create list that will hold class weights
    for moa in training_set.moa.unique():       # for each moa
        class_weights.append(counts[moa])  # add counts to class weights
    class_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device)
    return class_weights

def dict_splitting_into_tensor(df):
    '''
    Takes a dataframe and splits it into a dictionary of tensors.
    Input:
        df: pandas dataframe
    Output:
        dicti: dictionary of tensors (keys: moa string name, values: tensors)
    '''
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df["moa"].unique().reshape(-1,1))          # fit the encoder to the unique values of the moa column
    one_hot_encoded = enc.transform(enc.categories_[0].reshape(-1,1)).toarray()
    dicti = {}
    for i in range(0, len(enc.categories_[0])):       # create a dictionary with the one hot encoded values
        dicti[str(enc.categories_[0][i])] = one_hot_encoded[i]
    return dicti


def splitting(df):
    '''Splitting data into two parts:
    input : the pointer showing where the transcriptomic profile is  
    target : labels (the correct MoA)
    
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

# --------------------------------- Fine Tuning -----------------------------------------#
#----------------------------------------------------------------------------------------#
def set_parameter_requires_grad(model, feature_extracting):
    '''
    Sets the parameters of the model to require gradients or not.
    Input:
        model: model to be trained (pytorch model)
        feature_extracting: boolean value to determine if the model is being fine tuned or feature extracted
    Output:

    '''
    
    if feature_extracting:
        print("feature extracting in progress")
        for i, param in enumerate(model.parameters()):
            if i >= len(list(model.parameters())) - 2:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print("fine_tuning in progress")
        for param in model.parameters():
            param.requires_grad = True

# -------------------------Pre-Processing Used for DeepInsight -------------------------#
#----------------------------------------------------------------------------------------#
class LogScaler:
    """Log normalize and scale data

    Log normalization and scaling procedure as described as norm-2 in the
    DeepInsight paper supplementary information.
    
    Note: The dimensions of input matrix is (N samples, d features)
    """
    def __init__(self):
        self._min0 = None
        self._max = None

    """
    Use this as a preprocessing step in inference mode.
    """
    def fit(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

    """
    For training set only.
    """
    def fit_transform(self, X, y=None):
        # Min. of training set per feature
        self._min0 = X.min(axis=0)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Global max. of training set from X_norm
        self._max = X_norm.max()

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)

    """
    For validation and test set only.
    """
    def transform(self, X, y=None):
        # Adjust min. of each feature of X by _min0
        for i in range(X.shape[1]):
            X[:, i] = X[:, i].clip(min=self._min0[i], max=None)

        # Log normalized X by log(X + _min0 + 1)
        X_norm = np.log(
            X +
            np.repeat(np.abs(self._min0)[np.newaxis, :], X.shape[0], axis=0) +
            1).clip(min=0, max=None)

        # Normalized again by global max. of training set
        return (X_norm / self._max).clip(0, 1)
    
# ---------------------------------- Extracting Selected Transcriptomic profiles from .gctx file  ---------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------ #


def tprofiles_gc_too_func(data, clue_gene):
    '''
    Function preparing the gctoo dataframe to extract from gctx file, choosing only landmark genes
    
    Input:
        data: column meta data from clue.io that only includes training/test data
        clue_gene: row meta data from clue.io transcriptomic profiles
    
    Output:
        parsed gctoo file with all of the transcriptomic profiles. Only landmark genes included.'''

    clue_gene["gene_id"] = clue_gene["gene_id"].astype(str)
    landmark_gene_row_ids = clue_gene["gene_id"][clue_gene["feature_space"] == "landmark"]

    # get all samples (across all cell types, doses, and other treatment conditions) with certain MoA
    profile_ids = data["sig_id"]
    tprofiles_gctoo = parse("/scratch2-shared/erikep/level5_beta_trt_cp_n720216x12328.gctx", 
                                    cid= profile_ids, 
                                    rid = landmark_gene_row_ids)

    return tprofiles_gctoo

def extract_tprofile(profiles_gc_too, idx):
    '''Returns transcriptomic profile of of specific ID with in the form of a numpy array
    
    Input:
        profiles_gc_too: gc_too dataframe hosting transcriptomic profiles
        idx:  extract unique column name from L1000 data
    
    Output: 
        numpy array of a single transcriptomic profile
    '''
    tprofile_id =  profiles_gc_too.col_metadata_df.iloc[idx]
    tprofile_id_sig = [tprofile_id.name] 
    tprofile_gctoo = sg.subset_gctoo(profiles_gc_too, cid= tprofile_id_sig) 
    #return torch.tensor(tprofile_gctoo.data_df.values.astype(np.float32)) 
    return tprofile_id_sig, np.asarray(tprofile_gctoo.data_df.values.astype(np.float32))    


def np_array_transform(profiles_gc_too):
    '''
    Takes a .gctoo and extracts the correct profile, transforms the profile into a numpy array and then places it into a pandas data_frame.

    Input:
        profiles_gc_too: the gc_too dataframe with all the transcriptomic profiles

    Output:
        df: pandas dataframe, where each row is a transcriptomic profile
    '''
    rows = []
    sig_id_check = []
    for i in range(profiles_gc_too.data_df.shape[1]):
        sig_id_row, np_array = extract_tprofile(profiles_gc_too, i)
        rows.append(np_array)
        sig_id_check.append(sig_id_row)
    np_array =  np.asarray(rows)
    np_array = np_array.squeeze()
    df = pd.DataFrame(np_array)
    sig_id_df = pd.DataFrame(sig_id_check)
    df["sig_id"] =  sig_id_df[0:]
    return df

def acquire_npy(dataset, subset_data):
    '''
    Acquiring the numpy dataset in the npy format if it has already been created. Purpose is to save the reloading of the .npy dataframe, which can take 
    up to 9 minutes for the 10 MoAs.

    Input: 
    String with either "train" or "val". Then the user than manually inputs the name of the file.

    Ouput:
    Returns pandas dataframe from the .npy file found in '/scratch2-shared' given by the user.
    '''
    path = '/scratch2-shared/erikep/data_splits_npy'
    if dataset == 'train':
        #filename = input('Give name of npy file (str): ')
        #npy_set = np.load(path + filename)
        #npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_my10_train.npy', allow_pickle=True)
        #npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_cyc_adrtrain.npy', allow_pickle=True)
        if subset_data == "erik10":            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_train.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_train.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_8_12_train.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_8_12_train.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/tian10_train.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_adr_train.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_dop_train.npy', allow_pickle=True)
        else: 
            raise ValueError("subset_data must be either erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, tian10, cyc_adr or cyc_dop")  
        
    elif dataset == 'val':
        if subset_data == "erik10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_val.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_val.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_8_12_val.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_8_12_val.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/tian10_val.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_adr_val.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_dop_val.npy', allow_pickle=True)
        else: 
            raise ValueError("subset_data must be either erik10, tian10, cyc_adr or cyc_dop")  
    
    elif dataset == 'test':
        if subset_data == "erik10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_test.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_test.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_8_12_test.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_hq_8_12_test.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/tian10_test.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_adr_test.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/cyc_dop_test.npy', allow_pickle=True)
        else: 
            raise ValueError("subset_data must be either erik10, tian10, cyc_adr or cyc_dop")  
    else:
        filename =  input('Give name of npy file (str): ')
        npy_set = np.load(path + filename)
    df = pd.DataFrame(npy_set)
    return df.set_axis([*df.columns[:-1], 'sig_id'], axis=1, inplace=False)

def save_tprofile_npy(dataset, split_type, data_subset):
    '''Save the numpy array of the selected transcriptomics profiles
    Input:
        dataset: the numpy array to be saved
    '''
    path = '/scratch2-shared/erikep/data_splits_npy/'
    file_name = data_subset
    np.save(path + '_' + file_name + '_' + split_type, dataset)

def pre_processing(L1000_training, L1000_validation, L1000_test, 
         clue_gene, 
         npy_exists = True,
         use_variance_threshold = 0, 
         normalize = False, 
         save_npy = False,
         data_subset = "erik10"):
    '''
    Pre-processing of the data. The data is shuffled, the transcriptomic profiles are extracted, and transformed into numpy arrays, pre-processed
    all irrelevant columns are removed.

    Input:
        L1000_training: the training data
        L1000_validation: the validation data
        L1000_test: the test data
        clue_gene: genes that are used to extract the transcriptomic profiles
        npy_exists: boolean, if the numpy array has already been created, extract it from the npy file
        use_variance_threshold: if the user wants to use the variance threshold, the user can input a value between 0 and 1. 
                                If the user does not want to use the variance threshold, the user can input 0.
        normalize: boolean, if the user wants to normalize the data, the user can input True. If the user does not want
                             to normalize the data, the user can input False.  
        save_npy: boolean, if the user wants to save the numpy array, the user can input True.    
    Output:
        df_train_features: numpy array with the transcriptomic profiles of the training data
        df_val_features: numpy array with the transcriptomic profiles of the validation data
        df_train_labels: numpy array with the labels of the training data
        df_val_labels: numpy array with the labels of the validation data
        df_test_features: numpy array with the transcriptomic profiles of the test data
        df_test_labels: numpy array with the labels of the test data
        dict_moa: dictionary with moa as key (string) and one-hot encoded labels as values (numpy array)
        
    '''
    print("pre-processing data!")
    # shuffling training and validation data
    L1000_training = L1000_training.sample(frac = 1, random_state = 4)
    L1000_validation = L1000_validation.sample(frac = 1, random_state = 4)
    L1000_test = L1000_test.sample(frac = 1, random_state = 4)
    
    dict_moa = dict_splitting_into_tensor(L1000_training)
    print("extracting training transcriptomes")
    profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
    if npy_exists:
        df_train = acquire_npy('train', data_subset)
    else:    
        df_train = np_array_transform(profiles_gc_too_train)
        if save_npy:
            save_tprofile_npy(df_train, 'train', data_subset)

    print("extracting validation transcriptomes") 
    profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
    if npy_exists:
        df_val = acquire_npy('val', data_subset)
    else:    
        df_val = np_array_transform(profiles_gc_too_valid)
        if save_npy:
            save_tprofile_npy(df_val, 'val', data_subset)
    
    print("extracting test transcriptomes")
    profiles_gc_too_test = tprofiles_gc_too_func(L1000_test, clue_gene)
    if npy_exists:
        df_test = acquire_npy('test', data_subset)
    else:    
        df_test = np_array_transform(profiles_gc_too_test)
        if save_npy:
            save_tprofile_npy(df_test, 'test', data_subset)
    
    
    # merging the transcriptomic profiles with the corresponding MoA class using the sig_id as a key
    df_train = pd.merge(df_train, L1000_training[["sig_id", "Compound ID", "moa"]], how = "outer", on ="sig_id")
    df_val = pd.merge(df_val, L1000_validation[["sig_id", "Compound ID", "moa"]], how = "outer", on ="sig_id")
    df_test = pd.merge(df_test, L1000_test[["sig_id", "Compound ID", "moa"]], how = "outer", on ="sig_id")
   
   # dropping the sig_id column
    df_train.drop(columns = ["sig_id"], inplace = True)
    df_val.drop(columns = ["sig_id"], inplace = True)
    df_test.drop(columns = ["sig_id"], inplace = True)

     # separating the features from the labels
    df_train_features = df_train[df_train.columns[: -2]]
    df_val_features = df_val[df_val.columns[: -2]]
    df_train_labels = df_train[df_train.columns[-2:]]
    df_val_labels = df_val[df_val.columns[-2:]]
    df_test_features = df_test[df_test.columns[: -2]]
    df_test_labels = df_test[df_test.columns[-2:]]

    #pre-processing the data
    df_train_features, df_val_features, df_test_features = variance_threshold(df_train_features, df_val_features, use_variance_threshold, df_test_features)
    if normalize:
        df_train_features, df_val_features, df_test_features, norm_method = normalize_data(df_train_features, df_val_features, df_test_features)

    return df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa

def choose_cell_lines_to_include(df_set, clue_sig_in_SPECS, cell_lines):
    '''
    Returns training/validation/test set with only the cell lines specified by the user
    Input:
        df_set: a dataframe with the training/validation/test data
        cell_lines: a dictionary, where the key is the name of the moa and value is a list with the names of cell lines to be included.
            Default is empty. Ex. "{"cyclooxygenase inhibitor": ["A375", "HA1E"], "adrenergic receptor antagonist" : ["A375", "HA1E"] }"
    Output:
        df_set: a dataframe with the training/validation/test data with only the cell lines specified by the user
    '''
    profile_ids = clue_sig_in_SPECS[["Compound ID", "sig_id", "moa", 'cell_iname']][clue_sig_in_SPECS["Compound ID"].isin(df_set["Compound_ID"].unique())]
    return profile_ids[profile_ids["cell_iname"].isin(cell_lines)]

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


def create_splits(train, val, test, hq = "False", dose = "False", cell_lines = []):
    '''
    Input:
        train: a dataframe with training data
        val: a dataframe with validation data
        test: a dataframe with test data
        cc_q75: Threshold for 75th quantile of pairwise spearman correlation for individual, level 4 profiles.
        cell_lines: a dictionary, where the key is the name of the moa and value is a list with the names of cell lines to be included.
            Default is empty. Ex. "{"cyclooxygenase inhibitor": ["A375", "HA1E"], "adrenergic receptor antagonist" : ["A375", "HA1E"] }"+
    Output:
       Three pandas dataframes (for the different sets) with only necessary information
            - Compound ID
            - sig_id
            - moa
            - cell line
            - batch ID
    '''

    # read in documents
    clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")
    
    # Removing transcriptomic profiles based on the correlation of the level 4 profiles
    if hq == "True":
        clue_sig_in_SPECS = clue_sig_in_SPECS[clue_sig_in_SPECS["is_hiq"] == 1]

    if dose == "True":
        clue_sig_in_SPECS = clue_sig_in_SPECS[clue_sig_in_SPECS.pert_dose.between(8, 12)]
    else:
        clue_sig_in_SPECS = clue_sig_in_SPECS[clue_sig_in_SPECS.pert_dose.between(0, 100)]
    
    list_with_ans = []
    for set_split in [train, val, test]:
    # Removing transcriptomic profiles based on the correlation between different cell lines
        # Cell_Lines_Deprecated
        if len(cell_lines) > 0:
            profile_ids = choose_cell_lines_to_include(set_split, clue_sig_in_SPECS, cell_lines)
        else:
            profile_ids = clue_sig_in_SPECS[["Compound ID", "sig_id", "moa", 'cell_iname']][clue_sig_in_SPECS["Compound ID"].isin(set_split["Compound_ID"].unique())]
        list_with_ans.append(profile_ids)
    return list_with_ans[0], list_with_ans[1], list_with_ans[2]

# -------------------------------------------- Processing Chemical Structure Smiles Strings --------------------------------------------# 
# -------------------------------------------------------------------------------------------------------------------------------------#
# A function changing SMILES to Morgan fingerprints 
def smiles_to_array(smiles):
    '''
    Input:
        smiles: a string of a SMILES representation of a chemical structure
    Output:
        x_array: a numpy array of the Morgan fingerprint of the chemical structure
    '''
    molecules = Chem.MolFromSmiles(smiles) 
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(molecules, 2)
    x_array = []
    arrays = np.zeros(0,)
    DataStructs.ConvertToNumpyArray(fingerprints, arrays)
    x_array.append(arrays)
    x_array = np.asarray(x_array)
    x_array = ((np.squeeze(x_array)).astype(int)) 
    x_array = torch.from_numpy(x_array)
    return x_array    

# -------------------------------------------- Processing Cell Painting Images --------------------------------------------# 
# ---------------------------------------------------------------------------- --------------------------------------------#
def image_normalization(image, channel, plate, pd_image_norm):
    '''
    Normalizes the image by the mean and standard deviation 
    Pseudocode:
    1. using plate and channel, extract mean and standard deviation from pd_imgnorm
    2. normalize image used mean and std
    3. return normalized image
    '''

    if channel == "C1":
        extract = plate
    elif channel == "C2":
        extract = plate + '.1'
    elif channel == "C3":
        extract = plate + '.2'
    elif channel == "C4":
        extract = plate + '.3'
    else:
        extract = plate + '.4'
    single_cha = pd_image_norm[extract]
    
    mean = float(single_cha.iloc[1])
    std = float(single_cha.iloc[2])
    im_np =  (image - mean) / std
    return im_np

def channel_5_numpy_CID(df, CID, pd_image_norm):
    '''
    Puts together all channels from CP imaging into a single 5 x 256 x 256 tensor (c x h x w) from all_data.csv.
    Randomly samples a CP image from the rows that have the same compound name given the transcriptomic profile index
    that we are iterating through.
    Input
        df: file which contains all rows of image data with compound information (type = csv)
        CID: compound ID (type = string)
        pd_imgnorm: file which contains the mean and standard deviation for each channel (type = csv)
    
    Output:
        image: a single 5 x 256 x 256 tensor (c x h x w); randomly taken from the rows that have the same compound name
    '''
    # extract row with compound Name
    # This is currently not an ideal solution bc Compound_ID is not unique.
    # Will be fixed with batch ID, but this means paths document will need to be updated
    row = df[df["compound"] == CID]

    # randomly sample a CP image  from the rows that have the same compound name
    if row.shape[0] > 1:
        row = row.sample(n=1)
        
    # loop through all of the channels and add to single array
    im_list = []
    for c in range(1, 6):
        # extract by adding C to the integer we are looping
        #row_channel_path = row["C" + str(c)]
        local_im = cv2.imread(row["C" + str(c)].values[0], -1) # row.path would be same for me, except str(row[path]))
        
        # directly resize down to 256 by 256
        local_im = cv2.resize(local_im, (256, 256), interpolation = cv2.INTER_LINEAR)
        local_im = local_im.astype(np.float32)
        local_im_norm = image_normalization(local_im, c, row['plate'], pd_image_norm)
        # adds to array to the image vector 
        im_list.append(local_im_norm)
    
    arr_stack = np.stack(im_list, axis=0)
    # once we have all the channels, we covert it to a np.array, transpose so it has the correct dimensions and change the type for some reason
    #im = np.array(im).astype("int16")
    five_chan_img = torch.from_numpy(arr_stack)
    return five_chan_img

def channel_5_numpy(df, idx, pd_image_norm):
    '''
    Puts together all channels from CP imaging into a single 5 x 256 x 256 tensor (c x h x w) from all_data.csv
    Input
        df: file which contains all rows of image data with compound information (type = csv)
        idx: the index of the row (type = integer)
    
    Output:
        image: a single 5 x 256 x 256 tensor (c x h x w)
    '''
    # extract row with index 
    row = df.iloc[idx]
    
    # loop through all of the channels and add to single array
    im_list = []
    for c in range(1, 6):
        # extract by adding C to the integer we are looping
        #row_channel_path = row["C" + str(c)]
        local_im = cv2.imread(row["C" + str(c)], -1) # row.path would be same for me, except str(row[path]))
        local_im_norm = image_normalization(local_im, c, row['plate'], pd_image_norm)
        # directly resize down to 256 by 256
        local_im_norm = cv2.resize(local_im_norm, (256, 256), interpolation = cv2.INTER_LINEAR)
        local_im_norm = local_im_norm.astype(np.float32)
   
        # adds to array to the image vector 
        im_list.append(local_im_norm)
    
    arr_stack = np.stack(im_list, axis=0)
    # once we have all the channels, we covert it to a np.array, transpose so it has the correct dimensions and change the type for some reason
    #im = np.array(im).astype("int16")
    five_chan_img = torch.from_numpy(arr_stack)
    return five_chan_img

def accessing_correct_fold_csv_files(file):
 # download csvs with all the data pre split
    if file == 'tian10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/tian10/'
        train_filename = 'tian10_clue_train_fold_0.csv'
        val_filename = 'tian10_clue_val_fold_0.csv'
        test_filename = 'tian10_clue_test_fold_0.csv'
    elif file == 'erik10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_train_fold_0.csv'
        val_filename = 'erik10_clue_val_fold_0.csv'
        test_filename = 'erik10_clue_test_fold_0.csv'
    elif file == 'erik10_hq':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_hq_train_fold_0.csv'
        val_filename = 'erik10_clue_hq_val_fold_0.csv'
        test_filename = 'erik10_clue_hq_test_fold_0.csv'
    elif file == 'erik10_8_12':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_8_12__train_fold_0.csv'
        val_filename = 'erik10_clue_8_12__val_fold_0.csv'
        test_filename = 'erik10_clue_8_12__test_fold_0.csv'
    elif file == 'erik10_hq_8_12':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_hq_8_12__train_fold_0.csv'
        val_filename = 'erik10_clue_hq_8_12__val_fold_0.csv'
        test_filename = 'erik10_clue_hq_8_12__test_fold_0.csv'
    elif file == 'cyc_adr':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_adr/'
        train_filename = 'cyc_adr_clue_train_fold_0.csv'
        val_filename = 'cyc_adr_clue_val_fold_0.csv'
        test_filename = 'cyc_adr_clue_test_fold_0.csv'
    elif file == 'cyc_dop':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_dop/'
        train_filename = 'cyc_dop_clue_train_fold_0.csv'
        val_filename = 'cyc_dop_clue_val_fold_0.csv'
        test_filename = 'cyc_dop_clue_test_fold_0.csv'
    else:
        raise ValueError('Please enter a valid file name')

    training_set, validation_set, test_set =  load_train_valid_data(dir_path, train_filename, val_filename, test_filename)
    return training_set, validation_set, test_set

def checking_veracity_of_data(file, L1000_training, L1000_validation, L1000_test):
    """
    This function checks the number of profiles and unique compounds in the data set. 
    Is a sanity check to make sure the data is being loaded correctly.
    Asserts error if there is a discrepancy between unique compound number or transcriptomic profile number
    and the combined datasets-

    Inputs:
        file: string, name of the data set being investigated
        L1000_training: pandas dataframe, training set
        L1000_validation: pandas dataframe, validation set
        L1000_test: pandas dataframe, test set
    
    Returns:
        None

    Notice that the minus two is to take into account for enantiomers that Compound ID alone cannot distinguish between.
    """
    all_profiles = pd.concat([L1000_training, L1000_validation, L1000_test])
    if file == 'tian10':
        assert all_profiles.shape[0] == 13460, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == 121, 'Incorrect number of unique compounds'
    elif file == 'erik10':
        assert all_profiles.shape[0] == 18042, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(243 -2), 'Incorrect number of unique compounds'
    elif file == 'erik10_hq':
        assert all_profiles.shape[0] == 4564, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(185 -2), 'Incorrect number of unique compounds'
    elif file == 'erik10_8_12':
        assert all_profiles.shape[0] == 8644, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(238 - 2), 'Incorrect number of unique compounds'
    elif file == 'erik10_hq_8_12':
        assert all_profiles.shape[0] == 2387, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == 177, 'Incorrect number of unique compounds'
    elif file == 'cyc_adr':
        assert all_profiles.shape[0] == 1619, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(76 - 1), 'Incorrect number of unique compounds'
    elif file == 'cyc_dop':
        assert all_profiles.shape[0] == 3385, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(76 -1), 'Incorrect number of unique compounds'
    else:
        raise ValueError('Please enter a valid file name')
    print("passed veracity test!")

def label_smoothing(labels, smooth_factor, num_classes):
    labels = (1 - smooth_factor) * labels + smooth_factor / num_classes
    return labels

def create_terminal_table(elapsed_time, all_labels, all_predictions):
    '''Creates table for terminal output displaying test set accuracy and F1 score'''
    table = [["Time to Run Program", elapsed_time],
    ['Accuracy of Test Set', accuracy_score(all_labels, all_predictions)],
    ['F1 Score of Test Set', f1_score(all_labels, all_predictions, average='macro')]]
    print(tabulate(table, tablefmt='fancy_grid'))

def neptune_uploading(neptune_project_name,
                    file_name, 
                    model_name,
                    normalize,
                    yn_class_weights, 
                    learning_rate, 
                    elapsed_time, 
                    num_epochs,
                    loss_fn,
                    all_labels,
                    all_predictions,
                    model_data_subset,
                    dict_moa,
                    val_vs_train_loss_path,
                    val_vs_train_acc_path,
                    class_info_path,
                    conf_matrix_path,
                    model_path,
                    variance_thresh = False,
                    pixel_size = False,
                    loss_fn_train = False):
    '''
    This function uploads the results of the model to Neptune.ai
    '''
    run = neptune.init_run(project= neptune_project_name, api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
    run['model'] = model_name
    #run["feat_selec/feat_sel"] = feat_sel
    run["filename"] = file_name
    # uploading parameters
    run['parameters/normalize'] = normalize
    run['parameters/variance_threshold'] = variance_thresh
    run['parameters/class_weight'] = yn_class_weights
    run['parameters/learning_rate'] = learning_rate
    run['parameters/loss_fn_train'] = str(loss_fn_train)
    run['parameters/loss_function'] = str(loss_fn)
    run['parameters/pixel_size'] = pixel_size
    
    # Uploading validation metrics
    state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name)
    run['metrics/f1_score'] = state["f1_score"]
    run['metrics/accuracy'] = state["accuracy"]
    run['metrics/loss'] = state["valid_loss"]

    # length of time to train model
    run['metrics/time'] = elapsed_time
    run['metrics/epochs'] = num_epochs

    # Uploading test Metrics
    run['metrics/test_f1'] = f1_score(all_labels, all_predictions, average='macro')
    run['metrics/test_accuracy'] = accuracy_score(all_labels, all_predictions)

    conf_matrix_and_class_report(all_labels, all_predictions, model_data_subset, dict_moa)

    # Upload loss and accuracy plots
    run["images/loss"].upload(val_vs_train_loss_path)
    run["images/accuracy"].upload(val_vs_train_acc_path) 

    # Upload classification info
    run["files/classification_info"].upload(class_info_path)
    run["images/Conf_matrix.png"].upload(conf_matrix_path)

    # Upload model
    run["optimal_model_checkpoint"].upload(model_path)





