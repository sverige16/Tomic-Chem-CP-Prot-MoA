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
import re
import heapq


# ----------------------------------------- Visualization of Model Results ----------------------------------------------# 
# -----------------------------------------------------------------------------------------------------------------------#
def val_vs_train_loss(num_epochs, train_loss, val_loss, now, model_name, file_name, loss_path_to_save):
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
    x_axis = list(range(1, num_epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_loss, label = "train_loss")
    plt.plot(x_axis, val_loss, label = "val_loss")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss')
    plt.title(f'Validation versus Training Loss: {model_name + " " + file_name}')
    plt.legend()
    # plot
    plt.savefig(loss_path_to_save + '/' + 'loss_train_val_' + model_name + '_' + file_name + '_' + now  + '.png')
    return loss_path_to_save + '/' + 'loss_train_val_' + model_name + '_' + file_name + '_' + now   + '.png'

def val_vs_train_accuracy(num_epochs, train_acc, val_acc, now, model_name, file_name, acc_path_to_save):
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
    x_axis = list(range(1, num_epochs +1)) # create x axis with number of
    plt.plot(x_axis, train_acc, label = "train_acc")
    plt.plot(x_axis, val_acc, label = "val_acc")
    # Figure description
    plt.xlabel('# of Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Validation versus Training Accuracy: {model_name + " " + file_name}')
    plt.legend()
    # plot
    plt.savefig(acc_path_to_save + '/' + 'acc_train_val_' + model_name + '_' + file_name + '_' + now + '.png')
    return acc_path_to_save + '/' + 'acc_train_val_' + model_name + '_' + file_name + '_' + now + '.png'

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
        if key == 'adrenergic receptor agonist':
            list_of_MoA_names[np.argmax(value)] = 'adreAg'
        elif key == 'adrenergic receptor antagonist':
            list_of_MoA_names[np.argmax(value)] = 'adreAt'
        else:
            list_of_MoA_names[np.argmax(value)] = key[0:6]
        #list_of_MoA_names[np.argmax(value)] = key
    cf_matrix = confusion_matrix(labels_val, predictions)
    print(f' Confusion Matrix: {cf_matrix}')
    fig, ax = plt.subplots(figsize=(12, 10))  # Set the figure size
    sns.heatmap(cf_matrix, annot=True, fmt='d', ax=ax).set(title=f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)
    ax.set_xticklabels(list_of_MoA_names, rotation=45)
    ax.set_yticklabels(list_of_MoA_names, rotation=0)
    plt.tight_layout()  # Adjust layout to fit labels properly
    plt.savefig("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/conf_matrix")


   
    class_report = classification_report(labels_val, predictions, target_names= list_of_MoA_names)
    print(class_report)
    f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/class_info.txt","w")
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

def apply_class_weights_CL(training_set, dict_moa, device):
    """
    Applies class weights to the training set at the compound level. That is, the more compounds in a 
    class, the less weight it will have.
      --> This works well for Cell Painting, where for each compound, the same number of images are 
      taken. However, for the transcriptomic profiles, we have to balance according to the number of 
      transcriptomic profiles instead.
    Input:
        training_set: training set (pandas dataframe)
        device: device to run the model on (string)
    Output:
        class_weights: class weights (torch tensor)
    
    Note!: Do not use for BCG logit loss
    """
    counts = training_set.moa.value_counts()  # count the number of moa in each class for the ENTiRE dataset
    class_weights = len(dict_moa)*[0]   # create list that will hold class weights
    for key in dict_moa: 
              # for each moa
        number_of_compounds = counts[key]
        class_weights[np.argmax(dict_moa[key])] = 1/number_of_compounds
        #class_weights.append(counts[moa])  # add counts to class weights
    #lass_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device)
    return class_weights

def apply_class_weights_GE(train_np, training_set_split, dict_moa, device):
    """
    Applies class weights to the training set at the gene expression level. That is, the more compounds in a 
    class, the less weight it will have.
      --> This works well for Cell Painting, where for each compound, the same number of images are 
      taken. However, for the transcriptomic profiles, we have to balance according to the number of 
      transcriptomic profiles instead.
    Input:
        training_set: training set (pandas dataframe)
        device: device to run the model on (string)
    Output:
        class_weights: class weights (torch tensor)
    
    Note!: Do not use for BCG logit loss
    """
    train_np =  train_np.rename(columns ={train_np.columns[-1]: "sig_id"})
    train_np_moa = pd.merge(train_np, training_set_split, on = 'sig_id', how = 'inner')
    assert train_np_moa.shape[0] == train_np.shape[0]
    counts = train_np_moa.moa.value_counts()
    class_weights = len(dict_moa)*[0]   # create list that will hold class weights
    for key in dict_moa: 
              # for each moa
        number_of_tprofiles = counts[key]
        class_weights[np.argmax(dict_moa[key])] = 1/number_of_tprofiles
        #class_weights.append(counts[moa])  # add counts to class weights
    #lass_weights = [i / sum(class_weights) for  i in class_weights]
    class_weights= torch.tensor(class_weights,dtype=torch.float).to(device)
    return class_weights

def apply_class_weights_WRS(train_np, training_set_split, device):
    '''
    Used for Weighted Random Sampling
    Applies class weights to the training set at the transcriptomic profile level
    1. Merge transcriptomic profiles with chemical compound information 
    2. Create dictionary which stores moa as key and the 1/number of transcriptomics profiles as counts
    3. Associate a weight to each transcriptomic profile based on the moa
    
    Input:
        train_np: training set (pandas dataframe)
        training_set_split: training set split (pandas dataframe)
        device: device to run the model on (string)
    Output:
        samples_weight: class weights, weight for each sample in  (torch tensor)
    '''
    train_np =  train_np.rename(columns ={train_np.columns[-1]: "sig_id"})
    train_np_moa = pd.merge(train_np, training_set_split, on = 'sig_id', how = 'inner')
    assert train_np_moa.shape[0] == train_np.shape[0]
    counts = train_np_moa.moa.value_counts()
    class_weights = {}   # create list that will hold class weights
    for moa in train_np_moa.moa.unique():       # for each moa
        class_weights[moa] = 1/counts[moa]  # add counts to class weights
    samples_weight = torch.tensor([class_weights[t] for t in train_np_moa.moa])
    return samples_weight.to(device)
    

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
def set_parameter_requires_grad(model, feature_extracting, added_layers = 0):
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
            if i >= len(list(model.parameters())) - added_layers:
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
        if subset_data == "erik10":            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/erik10_train.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_train.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_8_12_train.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_8_12_train.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_tian10_train.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_adr_train.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_dop_train.npy', allow_pickle=True)
        else: 
            raise ValueError("subset_data must be either erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, tian10, cyc_adr or cyc_dop")  
        
    elif dataset == 'val':
        if subset_data == "erik10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_val.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_val.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_8_12_val.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_8_12_val.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_tian10_val.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_adr_val.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_dop_val.npy', allow_pickle=True)
        else: 
            raise ValueError("subset_data must be either erik10, tian10, cyc_adr or cyc_dop")  
    
    elif dataset == 'test':
        if subset_data == "erik10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_test.npy', allow_pickle=True)
        elif subset_data == "erik10_hq":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_test.npy', allow_pickle=True)
        elif subset_data == "erik10_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_8_12_test.npy', allow_pickle=True)
        elif subset_data == "erik10_hq_8_12":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_erik10_hq_8_12_test.npy', allow_pickle=True)
        elif subset_data == "tian10":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_tian10_test.npy', allow_pickle=True)
        elif subset_data == "cyc_adr":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_adr_test.npy', allow_pickle=True)
        elif subset_data == "cyc_dop":
            npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/_cyc_dop_test.npy', allow_pickle=True)
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
            x_train_var = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
            x_val_var = x_val.loc[:,var_thresh.variances_ > var_threshold]
            if x_train_var.shape[1] < x_train.shape[1]:
                print("Number of features removed due to Variance Threshold: ", x_train.shape[1] - x_train_var.shape[1])
            return x_train_var, x_val_var
        else:
            x_test_var = x_test.loc[:,var_thresh.variances_ > var_threshold]
            x_train_var = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
            x_val_var = x_val.loc[:,var_thresh.variances_ > var_threshold]
            if x_train_var.shape[1] < x_train.shape[1]:
                print("Number of features removed due to Variance Threshold: ", x_train.shape[1] - x_train_var.shape[1])

            return x_train_var, x_val_var, x_test_var
    
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
    df_train = pd.merge(df_train, L1000_training[["sig_id", "Compound ID", "moa"]], how = "inner", on ="sig_id")
    df_val = pd.merge(df_val, L1000_validation[["sig_id", "Compound ID", "moa"]], how = "inner", on ="sig_id")
    df_test = pd.merge(df_test, L1000_test[["sig_id", "Compound ID", "moa"]], how = "inner", on ="sig_id")
    
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
    if use_variance_threshold > 0:
        df_train_features, df_val_features, df_test_features = variance_threshold(df_train_features, df_val_features, use_variance_threshold, df_test_features)
        print("Variancethreshold used: ", use_variance_threshold)
    if normalize:
        df_train_features, df_val_features, df_test_features, norm_method = normalize_data(df_train_features, df_val_features, df_test_features)
        print("normalization method used: ", norm_method)
    return df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa


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

def dubbelchecking_image_normalization(pd_image_norm):
    # SPECS1
    # Used by Tian et al. to normalize the images for his project
    # SPECS2
    # Used by me to normalize the images for my project
    
    # SPECS3
    '''
    For every plate in pandas norm, check if the mean and std are the same for all channels in the imnorm that I am using
    '''
    
def apply_albumentations(image_array, transform):
    augmented = transform(image=image_array)
    return augmented['image']

def extracting_Tprofiles_with_cmpdID(tprofiles_df, split_sets, cmpdID, dict_cell_line):
    '''
    Extracts transcriptomic profiles with the same compound ID
    Input:
        df: file which contains all rows of transcriptomic profiles with compound information (type = csv)
        CID: compound ID (type = string)
    Output:
        df: file which contains all rows of transcriptomic profiles with compound information with the same compound ID (type = csv)
    '''
    matching_split_ids = split_sets["sig_id"][split_sets["Compound ID"] == cmpdID]
    random_single_sig_id = matching_split_ids.sample(n = 1)
    random_single_sig_id = random_single_sig_id.iloc[0]
    tprofile_with_sigid  = tprofiles_df[tprofiles_df[tprofiles_df.columns[-1]] == random_single_sig_id]
    tprofile = tprofile_with_sigid[tprofile_with_sigid.columns[0:-1]]
    tprofile = tprofile.to_numpy()
    cell_line_key = split_sets["cell_iname"][split_sets["sig_id"] == random_single_sig_id]
    cell_line_key = cell_line_key.iloc[0]
    t_cell_line = torch.tensor(dict_cell_line[cell_line_key])
    return torch.tensor(tprofile), t_cell_line.float()

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
        
       
        # Define the transformation: RandomCrop
        #crop_transform = RandomCrop(height=512, width=512)

         # Apply the transformation to the image array
        #local_im = apply_albumentations(local_im, crop_transform)
        #local_im = cv2.resize(local_im, (256, 256), interpolation = cv2.INTER_LINEAR)
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
        
        # Define the transformation: RandomCrop
        #crop_transform = RandomCrop(height=512, width=512)
        #local_im_norm = apply_albumentations(local_im_norm, crop_transform)
        # Downsizing image to 256 x 256
        local_im_norm = cv2.resize(local_im_norm, (256, 256), interpolation = cv2.INTER_LINEAR)
        local_im_norm = local_im_norm.astype(np.float32)
   
        # adds to array to the image vector 
        im_list.append(local_im_norm)
    
    arr_stack = np.stack(im_list, axis=0)
    # once we have all the channels, we covert it to a np.array, transpose so it has the correct dimensions and change the type for some reason
    #im = np.array(im).astype("int16")
    five_chan_img = torch.from_numpy(arr_stack)
    return five_chan_img


def accessing_all_folds_csv(file, fold_int):
    ''' 
    This functions returns the training, validation and test sets for a given file and fold number
    Input:
        file: the name of the file (type = string)
        fold_int: the fold number (type = int)
    Output:
        training_set: the training set (type = pandas dataframe)
        validation_set: the validation set (type = pandas dataframe)
        test_set: the test set (type = pandas dataframe)
    '''
    
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

    if file == 'tian10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/tian10/'
        if fold_int == 0:
            train_filename = 'tian10_clue_train_fold_0.csv'
            val_filename = 'tian10_clue_val_fold_0.csv'
            test_filename = 'tian10_clue_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'tian10_clue_train_fold_1.csv'
            val_filename = 'tian10_clue_val_fold_1.csv'
            test_filename = 'tian10_clue_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'tian10_clue_train_fold_2.csv'
            val_filename = 'tian10_clue_val_fold_2.csv'
            test_filename = 'tian10_clue_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'tian10_clue_train_fold_3.csv'
            val_filename = 'tian10_clue_val_fold_3.csv'
            test_filename = 'tian10_clue_test_fold_3.csv'
        else:
            train_filename = 'tian10_clue_train_fold_4.csv'
            val_filename = 'tian10_clue_val_fold_4.csv'
            test_filename = 'tian10_clue_test_fold_4.csv'
    elif file == 'tian10_all':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/tian10/'
        if fold_int ==0:
            train_filename = 'tian10_all_train_fold_0.csv'
            val_filename = 'tian10_all_val_fold_0.csv'
            test_filename = 'tian10_all_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'tian10_all_train_fold_1.csv'
            val_filename = 'tian10_all_val_fold_1.csv'
            test_filename = 'tian10_all_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'tian10_all_train_fold_2.csv'
            val_filename = 'tian10_all_val_fold_2.csv'
            test_filename = 'tian10_all_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'tian10_all_train_fold_3.csv'
            val_filename = 'tian10_all_val_fold_3.csv'
            test_filename = 'tian10_all_test_fold_3.csv'
        else:
            train_filename = 'tian10_all_train_fold_4.csv'
            val_filename = 'tian10_all_val_fold_4.csv'
            test_filename = 'tian10_all_test_fold_4.csv'
    elif file == 'erik10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        if fold_int ==0:
            train_filename = 'erik10_clue_train_fold_0.csv'
            val_filename = 'erik10_clue_val_fold_0.csv'
            test_filename = 'erik10_clue_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'erik10_clue_train_fold_1.csv'
            val_filename = 'erik10_clue_val_fold_1.csv'
            test_filename = 'erik10_clue_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'erik10_clue_train_fold_2.csv'
            val_filename = 'erik10_clue_val_fold_2.csv'
            test_filename = 'erik10_clue_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'erik10_clue_train_fold_3.csv'
            val_filename = 'erik10_clue_val_fold_3.csv'
            test_filename = 'erik10_clue_test_fold_3.csv'
        else:
            train_filename = 'erik10_clue_train_fold_4.csv'
            val_filename = 'erik10_clue_val_fold_4.csv'
            test_filename = 'erik10_clue_test_fold_4.csv'
    elif file == 'erik10_all':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        if fold_int ==0:
            train_filename = 'erik10_all_train_fold_0.csv'
            val_filename = 'erik10_all_val_fold_0.csv'
            test_filename = 'erik10_all_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'erik10_all_train_fold_1.csv'
            val_filename = 'erik10_all_val_fold_1.csv'
            test_filename = 'erik10_all_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'erik10_all_train_fold_2.csv'
            val_filename = 'erik10_all_val_fold_2.csv'
            test_filename = 'erik10_all_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'erik10_all_train_fold_3.csv'
            val_filename = 'erik10_all_val_fold_3.csv'
            test_filename = 'erik10_all_test_fold_3.csv'
        else:
            train_filename = 'erik10_all_train_fold_4.csv'
            val_filename = 'erik10_all_val_fold_4.csv'
            test_filename = 'erik10_all_test_fold_4.csv'
    elif file == 'erik10_hq':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        if fold_int ==0:
            train_filename = 'erik10_clue_hq_train_fold_0.csv'
            val_filename = 'erik10_clue_hq_val_fold_0.csv'
            test_filename = 'erik10_clue_hq_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'erik10_clue_hq_train_fold_1.csv'
            val_filename = 'erik10_clue_hq_val_fold_1.csv'
            test_filename = 'erik10_clue_hq_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'erik10_clue_hq_train_fold_2.csv'
            val_filename = 'erik10_clue_hq_val_fold_2.csv'
            test_filename = 'erik10_clue_hq_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'erik10_clue_hq_train_fold_3.csv'
            val_filename = 'erik10_clue_hq_val_fold_3.csv'
            test_filename = 'erik10_clue_hq_test_fold_3.csv'
        else:
            train_filename = 'erik10_clue_hq_train_fold_4.csv'
            val_filename = 'erik10_clue_hq_val_fold_4.csv'
            test_filename = 'erik10_clue_hq_test_fold_4.csv'
    elif file == 'erik10_8_12':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        if fold_int ==0:
            train_filename = 'erik10_clue_8_12__train_fold_0.csv'
            val_filename = 'erik10_clue_8_12__val_fold_0.csv'
            test_filename = 'erik10_clue_8_12__test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'erik10_clue_8_12__train_fold_1.csv'
            val_filename = 'erik10_clue_8_12__val_fold_1.csv'
            test_filename = 'erik10_clue_8_12__test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'erik10_clue_8_12__train_fold_2.csv'
            val_filename = 'erik10_clue_8_12__val_fold_2.csv'
            test_filename = 'erik10_clue_8_12__test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'erik10_clue_8_12__train_fold_3.csv'
            val_filename = 'erik10_clue_8_12__val_fold_3.csv'
            test_filename = 'erik10_clue_8_12__test_fold_3.csv'
        else:
            train_filename = 'erik10_clue_8_12__train_fold_4.csv'
            val_filename = 'erik10_clue_8_12__val_fold_4.csv'
            test_filename = 'erik10_clue_8_12__test_fold_4.csv'
    elif file == 'erik10_hq_8_12':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        if fold_int ==0:
            train_filename = 'erik10_clue_hq_8_12__train_fold_0.csv'
            val_filename = 'erik10_clue_hq_8_12__val_fold_0.csv'
            test_filename = 'erik10_clue_hq_8_12__test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'erik10_clue_hq_8_12__train_fold_1.csv'
            val_filename = 'erik10_clue_hq_8_12__val_fold_1.csv'
            test_filename = 'erik10_clue_hq_8_12__test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'erik10_clue_hq_8_12__train_fold_2.csv'
            val_filename = 'erik10_clue_hq_8_12__val_fold_2.csv'
            test_filename = 'erik10_clue_hq_8_12__test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'erik10_clue_hq_8_12__train_fold_3.csv'
            val_filename = 'erik10_clue_hq_8_12__val_fold_3.csv'
            test_filename = 'erik10_clue_hq_8_12__test_fold_3.csv'
        else:
            train_filename = 'erik10_clue_hq_8_12__train_fold_4.csv'
            val_filename = 'erik10_clue_hq_8_12__val_fold_4.csv'
            test_filename = 'erik10_clue_hq_8_12__test_fold_4.csv'
    elif file == 'cyc_adr':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_adr/'
        if fold_int ==0:
            train_filename = 'cyc_adr_clue_train_fold_0.csv'
            val_filename = 'cyc_adr_clue_val_fold_0.csv'
            test_filename = 'cyc_adr_clue_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'cyc_adr_clue_train_fold_1.csv'
            val_filename = 'cyc_adr_clue_val_fold_1.csv'
            test_filename = 'cyc_adr_clue_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'cyc_adr_clue_train_fold_2.csv'
            val_filename = 'cyc_adr_clue_val_fold_2.csv'
            test_filename = 'cyc_adr_clue_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'cyc_adr_clue_train_fold_3.csv'
            val_filename = 'cyc_adr_clue_val_fold_3.csv'
            test_filename = 'cyc_adr_clue_test_fold_3.csv'
        else:
            train_filename = 'cyc_adr_clue_train_fold_4.csv'
            val_filename = 'cyc_adr_clue_val_fold_4.csv'
            test_filename = 'cyc_adr_clue_test_fold_4.csv'
    elif file == 'cyc_dop':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_dop/'
        if fold_int ==0:
            train_filename = 'cyc_dop_clue_train_fold_0.csv'
            val_filename = 'cyc_dop_clue_val_fold_0.csv'
            test_filename = 'cyc_dop_clue_test_fold_0.csv'
        elif fold_int == 1:
            train_filename = 'cyc_dop_clue_train_fold_1.csv'
            val_filename = 'cyc_dop_clue_val_fold_1.csv'
            test_filename = 'cyc_dop_clue_test_fold_1.csv'
        elif fold_int == 2:
            train_filename = 'cyc_dop_clue_train_fold_2.csv'
            val_filename = 'cyc_dop_clue_val_fold_2.csv'
            test_filename = 'cyc_dop_clue_test_fold_2.csv'
        elif fold_int == 3:
            train_filename = 'cyc_dop_clue_train_fold_3.csv'
            val_filename = 'cyc_dop_clue_val_fold_3.csv'
            test_filename = 'cyc_dop_clue_test_fold_3.csv'
        else:
            train_filename = 'cyc_dop_clue_train_fold_4.csv'
            val_filename = 'cyc_dop_clue_val_fold_4.csv'
            test_filename = 'cyc_dop_clue_test_fold_4.csv'
    else:
        raise ValueError('Please enter a valid file name')

    training_set, validation_set, test_set =  load_train_valid_data(dir_path, train_filename, val_filename, test_filename)
    return training_set, validation_set, test_set


def label_smoothing(labels, smooth_factor, num_classes):
    '''
    Applies label smoothing to labels
    Inputs:
        labels: tensor, labels
        smooth_factor: float, smoothing factor
        num_classes: int, number of target classes
    Returns:
        labels: tensor, smoothed labels
    '''
    labels = (1 - smooth_factor) * labels + smooth_factor / num_classes
    return labels

def create_terminal_table(elapsed_time, all_labels, all_predictions):
    '''Creates table for terminal output displaying test set accuracy and F1 score
    Inputs:
        elapsed_time: float, time to run program
        all_labels: list, list of all labels
        all_predictions: list, list of all predictions
    Returns:
        None
    '''
    table = [["Time to Run Program", elapsed_time],
    ['Accuracy of Test Set', accuracy_score(all_labels, all_predictions)],
    ['F1 Score of Test Set', f1_score(all_labels, all_predictions, average='macro')]]
    print(tabulate(table, tablefmt='fancy_grid'))

def upload_to_neptune(neptune_project_name,
                    file_name, 
                    model_name,
                    normalize,
                    yn_class_weights,  
                    elapsed_time, 
                    all_labels,
                    all_predictions,
                    dict_moa,
                    loss_fn = False,
                    num_epochs = False,
                    learning_rate = False,
                    val_vs_train_loss_path = False,
                    val_vs_train_acc_path = False,
                    variance_thresh = False,
                    pixel_size = False,
                    loss_fn_train = False, 
                    learning_rate_scheduler = 'default'):
    '''
    This function uploads the results of the model to Neptune.ai

    Inputs:
        neptune_project_name: string, name of the project in Neptune.ai
        file_name: string, name of the data set used
        model_name: string, name of the model used
        normalize: boolean, whether or not the data was normalized using QuauntileTransformer
        yn_class_weights: boolean, whether or not class weights were used
        learning_rate: float, initial learning rate used
        elapsed_time: float, time it took to run the model
        num_epochs: int, number of epochs used
        loss_fn: string, name of the loss function used
        all_labels: list, list of all the labels
        all_predictions: list, list of all the predictions
        dict_moa: dictionary, dictionary of the MoA classes associated with one hot encoding
        val_vs_train_loss_path: string, path to the plot of the validation vs training loss
        val_vs_train_acc_path: string, path to the plot of the validation vs training accuracy
        variance_thresh: float, variance threshold used for feature selection
        pixel_size: int, pixel size used for DeepInsight or Cell Painting
        loss_fn_train: string, name of the loss function used for training, specifically for label smoothing
    '''
    run = neptune.init_run(project= neptune_project_name, api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
    run['model'] = model_name
    run["filename"] = file_name
    model_data_subset = model_name + '_' + file_name

    # uploading parameters
    run['parameters/normalize'] = normalize
    run['parameters/variance_threshold'] = variance_thresh
    run['parameters/class_weight'] = yn_class_weights
    if learning_rate:
        run['parameters/learning_rate'] = learning_rate
    else:
        run['parameters/learning_rate'] = 'False'
    if loss_fn_train:
        run['parameters/loss_fn_train'] = str(loss_fn_train)
    else:
        run['parameters/loss_fn_train'] = 'False'
    if loss_fn:
        run['parameters/loss_function'] = str(loss_fn)
    else:
        run['parameters/loss_function'] = 'False'
    if learning_rate_scheduler:
        run['parameters/learning_rate_scheduler'] = learning_rate_scheduler
    else:
        run['parameters/learning_rate_scheduler'] = 'False'
    if pixel_size:
        run['parameters/pixel_size'] = pixel_size
    else:
        run['parameters/pixel_size'] = 'False'
    

    # length of time to train model
    run['metrics/time'] = elapsed_time
    if num_epochs:
        run['metrics/epochs'] = num_epochs
    else:
        run['metrics/epochs'] = 'False'

    # Uploading test Metrics
    run['metrics/test_f1'] = f1_score(all_labels, all_predictions, average='macro')
    run['metrics/test_accuracy'] = accuracy_score(all_labels, all_predictions)

    # Upload loss and accuracy plots
    if val_vs_train_loss_path:
        run["images/loss"].upload(val_vs_train_loss_path)
    if val_vs_train_acc_path:
        run["images/accuracy"].upload(val_vs_train_acc_path) 

    if val_vs_train_acc_path or val_vs_train_acc_path:
         # Upload model
        run["optimal_model_checkpoint"].upload('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')
         # Uploading validation metrics
        state = torch.load('/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt')
        run['metrics/f1_score'] = state["f1_score"]
        run['metrics/accuracy'] = state["accuracy"]

    conf_matrix_and_class_report(all_labels, all_predictions, model_data_subset, dict_moa)

        # Upload classification info
    run["files/classif_info"].upload('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/class_info.txt')
    run["images/conf_matrix"].upload('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/conf_matrix.png')
    run.stop()
    
def set_bool_hqdose(file_name):
    '''Sets boolean values for hq and dose
    Inputs:
        file_name: string, name of the data set used
    Returns:
        hq: boolean, whether or not the data set being used only has high quality profiles
        dose: boolean, whether or not the data set being used only has dose within 8 - 12'''
    hq, dose = 'False', 'False'
    if re.search('hq', file_name):
        hq = 'True'
    if re.search('8', file_name):
        dose = 'True'
    return hq, dose
    
def set_bool_npy(variance_thresh, normalize_c, five_fold = 'False'):
    '''Sets boolean values for npy_exists and save_npy
    Inputs:
        variance_thresh: float, variance threshold used for feature selection
        normalize_c: boolean, whether or not the data was normalized using QuantileTransformer
    Returns:
        npy_exists: boolean, whether or not the npy file exists
        save_npy: boolean, whether or not to save the npy file'''
    if variance_thresh > 0 or normalize_c == 'True' or five_fold == 'True':
        npy_exists = False
        save_npy = False    
    else:
        npy_exists = True
        save_npy = False

    return npy_exists, save_npy

def find_two_lowest_numbers(lst):
    '''Find the two lowest numbers in a list. Idea is to find the two lowest validation losses and
    divide by two to make sure some random drop in validation loss is not a better model
    than some other model that has a lower average validation loss.
    Input: 
        lst: list of numbers
    Output: 
        lowest1, lowest2: two lowest numbers in the list'''
    return heapq.nsmallest(2, lst)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = torch.log_softmax(inputs, dim=-1)

        if len(targets.shape) == 1:
            targets_one_hot = torch.zeros_like(log_probs)
            targets_one_hot.scatter_(1, targets.view(-1, 1), 1)
        else:
            targets_one_hot = targets

        # Compute the cross-entropy loss using log_probs and one-hot targets
        ce_loss = -torch.sum(targets_one_hot * log_probs, dim=-1)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
    
def differentiable_f1_loss(y_true, y_pred, class_weights=None):
    smooth = 1e-7

    # Convert predictions to binary by rounding
    y_pred = torch.round(torch.clamp(y_pred, 0, 1))

    # Calculate true positives, false positives, and false negatives
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    # Calculate continuous precision and recall
    p = tp / (tp + fp + smooth)
    r = tp / (tp + fn + smooth)

    # Calculate the differentiable F1 score
    f1 = 2 * p * r / (p + r + smooth)
    f1[torch.isnan(f1)] = 0

    # Apply class weights if provided
    if class_weights is not None:
        f1 = f1 * class_weights
        f1 /= torch.sum(class_weights)

    # Calculate the loss as 1 - F1 score
    return 1 - torch.mean(f1)

# ---------------------------------------- Training, Validation, Test Loops ------------------------#       
# ----------------------------------------- Training Loops ----------------------------------------- #    
def adapt_training_loop(n_epochs,
                        optimizer, 
                        model, 
                        loss_fn, 
                        loss_fn_str, 
                        train_loader, 
                        valid_loader,  
                        device, 
                        model_name, 
                        val_str, 
                        early_patience = 0, 
                        loss_fn_train = "false",
                        my_lr_scheduler = "false"):
    '''
  
    '''
    def best_val_metric(val_str):
        '''
        The initial validation metric number to compare and see if performance has improved
        Input:
            val_str: string, fe, f1, or loss
        Output:
            val_metric: float, 0 or np.inf depending on the validation metric'''
        if val_str == "fe":
            return 0
        elif val_str == "f1":
            return 0
        elif val_str == "loss":
            return np.inf
        else:
            ValueError("No validation metric identified")
    # lists keep track of loss and accuracy for training and validation set
    model = model.to(device)
    if early_patience > 0:
        early_stopper = EarlyStopper(patience=early_patience, min_delta=0.001)
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    val_f1_score_per_epoch = []
    val_metric = best_val_metric(val_str)
    if loss_fn_train != "false":
        loss_fn_train.train().to(device)
    outer_loop_done = False
    for epoch in tqdm(range(1, n_epochs +1), desc = "Epoch", position=0, leave= False):
        loss_train = 0.0
        train_total = 0
        train_correct = 0
        if outer_loop_done:
            break
        for data_labels in tqdm(train_loader, desc = "train batch", position=0, leave= False):
            optimizer.zero_grad()
            # put model, images, labels on the same device
            data = [d.to(device=device) for d in data_labels[:-1]]  # Excluding the labels from data
            labels = data_labels[-1].to(device=device)
            # Assessing outputs
            outputs = model(*data)
            outputs = outputs.to(device=device)
            loss = loss_fn_loop(loss_fn_train = loss_fn_train, 
                                loss_fn = loss_fn, 
                                loss_fn_str = loss_fn_str, 
                                outputs = outputs, 
                                labels = labels)
            #loss = loss_fn(outputs,labels)
            # For L2 regularization
            #l2_lambda = 0.000001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss = loss + l2_lambda * l2_norm
            # Update weights
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                outer_loop_done = True
                break
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # Training Metrics
            loss_train += loss.item()
            train_predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            train_total += labels.shape[0]
            train_correct += int((train_predicted == labels).sum())
        if loss_fn_train != "false":
            loss_fn_train.eval()
        # validation metrics from batch
        val_correct, val_total, val_loss, val_metric_upd, val_f1_score = adapt_validation_loop(model = model,
                                                                                                loss_fn = loss_fn,
                                                                                                loss_fn_train = loss_fn_train,
                                                                                                loss_fn_str = loss_fn_str, 
                                                                                                val_str = val_str, 
                                                                                                valid_loader = valid_loader, 
                                                                                                val_metric = val_metric, 
                                                                                                device = device, 
                                                                                                model_name = model_name)
        val_metric = val_metric_upd
        val_accuracy = val_correct/val_total
        # printing results for epoch
        print(f' Epoch: {epoch}, Training loss: {loss_train/len(train_loader)}, Validation Loss: {val_loss}, F1 Score: {val_f1_score} ')
        # adding epoch loss, accuracy to lists 
        val_loss_per_epoch.append(val_loss)
        train_loss_per_epoch.append(loss_train/len(train_loader))
        val_acc_per_epoch.append(val_accuracy)
        train_acc_per_epoch.append(train_correct/train_total)
        val_f1_score_per_epoch.append(val_f1_score)
        if loss_fn_train != "false":
            loss_fn_train.next_epoch()
        if early_patience > 0:
            if early_stopper.early_stop(validation_loss = val_loss):             
                break
        if my_lr_scheduler != "false":
            my_lr_scheduler.step()
    # return lists with loss, accuracy every epoch
    return train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, epoch

# ----------------------------------------- Validation Loops ----------------------------------------- #
def adapt_validation_loop(model, 
                        loss_fn,
                        loss_fn_str, 
                        loss_fn_train,
                        valid_loader, 
                        val_metric, 
                        device, 
                        model_name, 
                        val_str):
    '''
    Validation loop for the various different models. Can take various number of inputs.
    Input:
        model: model to be trained
        loss_fn: string, loss function to be used
        loss_fn_str: string, loss function to be used
        valid_loader: validation data loader
        val_metric: string, metric to be used for validation
        device: device to be used for training
        model_name: string, name of the model
        val_str: string, fe, f1, or loss
    Output:
        val_correct: int, number of correct predictions
        val_total: int, total number of predictions
        val_loss: float, loss of the model
        val_metric: float, metric of the model (f1 or loss)
        val_f1_score: float, f1 score of the model
    '''

    def validation_saving(val_str, 
                        val_metric, 
                        avg_val_loss, 
                        avg_f1, 
                        model_name, 
                        model, 
                        predictions, 
                        all_labels, 
                        predict_proba):
        '''
        Saving the model based on the validation metric
        Input:
            val_str: string, fe, f1, or loss
            val_metric: float, the validation metric
            avg_val_loss: float, the average validation loss across the epoch
            avg_f1: float, the average f1 score across the epoch
            model_name: string, the name of the model
            model: the model
            predictions: list, the predictions from the model
            all_labels: list, the labels from the model
            predict_proba: list, the predicted probabilities from the model
        Output:
            val_metric: float, the updated best score for the validation metric of choice
        '''
        if val_str == "fe":
            pass
        elif val_str == "f1":
            if val_metric < avg_f1:
                val_metric = avg_f1
                m = torch.nn.Softmax(dim=1)
                pred_cpu = torch.cat(predictions).cpu()
                labels_cpu =  torch.cat(all_labels).cpu()
                torch.save(
                    {   'predict_proba' : m(torch.cat(predict_proba)),
                        'predictions' : pred_cpu.numpy(),
                        'labels_val' : labels_cpu.numpy(),
                        'model_state_dict' : model.state_dict(),
                        'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                        'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
                },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
                )
        elif val_str == "val_loss":
            if val_metric > avg_val_loss:
                val_metric = avg_val_loss
                m = torch.nn.Softmax(dim=1)
                pred_cpu = torch.cat(predictions).cpu()
                labels_cpu =  torch.cat(all_labels).cpu()
                torch.save(
                    {   'predict_proba' : m(torch.cat(predict_proba)),
                        'predictions' : pred_cpu.numpy(),
                        'labels_val' : labels_cpu.numpy(),
                        'model_state_dict' : model.state_dict(),
                        'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                        'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
                },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
                )
        else:
            ValueError("No validation metric identified")
        return val_metric
    
    model.to(device)
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    predict_proba = []
    predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on validation data.
        for data_labels in tqdm(valid_loader, desc = "valid", position=0, leave= False):
            # put model, images, labels on the same device
            data = [d.to(device=device) for d in data_labels[:-1]]  # Excluding the labels from data
            labels = data_labels[-1].to(device=device)
            # Assessing outputs
            outputs = model(*data)
            outputs = outputs.to(device=device)
            loss = loss_fn_loop(loss_fn_train = loss_fn_train, loss_fn = loss_fn, loss_fn_str = loss_fn_str, outputs = outputs, labels = labels)
            loss_val += loss.item()
            predicted = torch.argmax(outputs, 1)
            labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum()) # saving best 
            all_labels.append(labels)
            predict_proba.append(outputs)
            predictions.append(predicted)
        avg_val_loss = loss_val/len(valid_loader)  # average loss over batch
        pred_cpu = torch.cat(predictions).cpu()
        labels_cpu =  torch.cat(all_labels).cpu()
        val_metric = validation_saving(val_str = val_str, 
                                       val_metric = val_metric, 
                                       predictions = predictions, 
                                       all_labels = all_labels, 
                                       predict_proba=predict_proba,
                                       avg_f1 = f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                                       avg_val_loss=avg_val_loss,
                                       model=model,
                                       model_name=model_name)
    model.train()
    return correct, total, avg_val_loss, val_metric,  f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro')

def adapt_test_loop(model, 
                    test_loader, 
                    device):
    '''
    Test loop for the various models. Can take several different inputs
    Input:
        model: model to be tested
        test_loader: test data loader
        device: device to run the model on
    '''
    model = model.to(device)
    model.eval()
    loss_test = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # does not keep track of gradients so as to not train on test data.
        for data_labels in tqdm(test_loader, desc = "test", position=0, leave= False):
            # put model, images, labels on the same device
            data = [d.to(device=device) for d in data_labels[:-1]]  # Excluding the labels from data
            labels = data_labels[-1].to(device=device)
            # Assessing outputs
            outputs = model(*data)
            predicted = torch.argmax(outputs, 1)
            #labels = torch.argmax(labels,1)
            total += labels.shape[0]
            correct += int((predicted == torch.max(labels, 1)[1]).sum())
            all_predictions = all_predictions + predicted.tolist()
            all_labels = all_labels + torch.max(labels, 1)[1].tolist()
        avg_test_loss = loss_test/len(test_loader)  # average loss over batch
    return correct, total, avg_test_loss, all_predictions, all_labels


############################################################################################################
def choose_device(using_cuda):
    '''
    Choosing device to train on (GPU or CPU)
    
    Input:
        using_cuda: boolean
    Output:
        device: torch.device
    '''
    if using_cuda:
        device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    else:
        device = torch.device('cpu')
    print(f'Training on device {device}. ' )
    return device

def different_loss_functions( loss_fn_str, 
                             loss_fn_train_str = 'false', 
                             n_classes = 10, class_weights = None,  
                            alpha = 0.5, gamma = 0.25 , smoothing = 0.1):
    '''
    Choosing loss function to train on, allows for the flexibility of using the same training/validation/test loops
        across different loss functions and different algorithms
    Input:
        loss_fn_str: string, loss function to train, validate and test on or only validate and test
        loss_fn_train_str: string, loss function to train on, else false
        n_classes: int, number of classes
        class_weights: list, weights for each class
        alpha: float, alpha parameter for focal loss
        gamma: float, gamma parameter for focal loss
        smoothing: float, smoothing parameter for label smoothing
    Output:
        loss_fn_train: loss function to train on, else false
        loss_fn: loss function to validate and test alternatively train, validate and test on
    '''
    if loss_fn_train_str ==  'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = alpha, n_classes=n_classes, smoothing = smoothing)
        if loss_fn_str == 'BCE':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_fn_str == 'cross':
            loss_fn = torch.nn.CrossEntropyLoss(weight = class_weights)
        elif loss_fn_str == 'focal':
            loss_fn = FocalLoss(gamma=gamma, alpha=alpha)
        else:
            ValueError('No loss function identified') 
        return loss_fn_train, loss_fn
    elif loss_fn_str == 'BCE':
        return 'false', torch.nn.BCEWithLogitsLoss()
    elif loss_fn_str == 'cross':
        return 'false', torch.nn.CrossEntropyLoss(weight = class_weights)
    elif loss_fn_str == 'focal':
        return 'false', FocalLoss(gamma=gamma, alpha=alpha)
    else:
        ValueError('No loss function identified')

class Transcriptomic_Profiles_gc_too(torch.utils.data.Dataset):
    '''
    Works with profiles_gc_too_func to create a dataset of transcriptomic profiles and labels
    '''
    def __init__(self, gc_too, split, dict_moa):
        #self.tprofile_labels = labels
        self.profiles_gc_too = gc_too
        self.split_sets = split
        self.dict_moa = dict_moa
        
    def __len__(self):
        ''' The number of data points '''
        return len(self.split_sets)

    def __getitem__(self, idx):
        '''Retreiving the transcriptomic profile and label
        Pseudocode:
        1. Extract the transcriptomic profile using the index along with sig_id
        2. Extract the label from t_profile
        3. Use sig_id to extract the label from split_sets
        4. Convert label to one hot encoding using function
        5. Convert to torch tensors and return'''

        t_profile = extract_tprofile(self.profiles_gc_too, idx)          # extract image from csv using index
        t_sig_id = t_profile[0][0]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        t_moa = torch.tensor(self.dict_moa[moa_key])
        t_profile_features = torch.tensor(t_profile[1])       # turn t profile into a floating torch tensor
        
        return torch.squeeze(t_profile_features), t_moa 

class Transcriptomic_Profiles_numpy(torch.utils.data.Dataset):
    '''
    Works with profiles_gc_too_func to create a dataset of transcriptomic profiles and labels
    '''
    def __init__(self, np_array, split, dict_moa):
        #self.tprofile_labels = labels
        self.profiles_np_array = np_array
        self.split_sets = split
        self.dict_moa = dict_moa
        
    def __len__(self):
        ''' The number of data points '''
        return len(self.split_sets)

    def __getitem__(self, idx):
        '''Retreiving the transcriptomic profile and label
        Pseudocode:
        1. Extract the transcriptomic profile using the index along with sig_id
        2. Extract the label from t_profile
        3. Use sig_id to extract the label from split_sets
        4. Convert label to one hot encoding using function
        5. Convert to torch tensors and return'''

        t_profile = self.profiles_np_array.iloc[idx, :-1]          # extract image from csv using index
        t_sig_id = self.profiles_np_array.iloc[idx, -1]
        moa_key = self.split_sets["moa"][self.split_sets["sig_id"] == t_sig_id]
        moa_key = moa_key.iloc[0]
        t_moa = torch.tensor(self.dict_moa[moa_key])
        t_profile_features = torch.tensor(t_profile)       # turn t profile into a floating torch tensor
        
        return t_profile_features, t_moa 


def loss_fn_loop(loss_fn_train, loss_fn, loss_fn_str, outputs, labels):
    '''
    Choosing loss function to train on, allows for the flexibility of using the same training/validation/test loops
        across different loss functions and different algorithms
    
    Input:
        loss_fn_str: string, if "false", not ols, otherwise ols
        loss_fn: loss function to validate and test alternatively train, validate and test on
    Output:
        loss: loss from output of model and labels
        '''
    if loss_fn_train != "false":
        loss = loss_fn_train(outputs, torch.max(labels, 1)[1])
    elif loss_fn_str == 'BCE' or loss_fn_str == 'focal':
        loss = loss_fn(outputs,labels)
    else:
        loss = loss_fn(outputs, torch.max(labels, 1)[1])
    return loss

def validation_saving(val_str, 
                      val_metric, 
                      avg_val_loss, 
                      avg_f1, 
                      model_name, 
                      model, 
                      predictions, 
                      all_labels, 
                      predict_proba):
    '''
    Saving the model based on the validation metric
    Input:
        val_str: string, fe, f1, or loss
        val_metric: float, the validation metric
        avg_val_loss: float, the average validation loss across the epoch
        avg_f1: float, the average f1 score across the epoch
        model_name: string, the name of the model
        model: the model
        predictions: list, the predictions from the model
        all_labels: list, the labels from the model
        predict_proba: list, the predicted probabilities from the model
    Output:
        val_metric: float, the updated best score for the validation metric of choice
    '''
    if val_str == "fe":
        pass
    elif val_str == "f1":
        if val_metric < avg_f1:
            val_metric = avg_f1
            m = torch.nn.Softmax(dim=1)
            pred_cpu = torch.cat(predictions).cpu()
            labels_cpu =  torch.cat(all_labels).cpu()
            torch.save(
                {   'predict_proba' : m(torch.cat(predict_proba)),
                    'predictions' : pred_cpu.numpy(),
                    'labels_val' : labels_cpu.numpy(),
                    'model_state_dict' : model.state_dict(),
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
            )
    elif val_str == "val_loss":
        if val_metric > avg_val_loss:
            val_metric = avg_val_loss
            m = torch.nn.Softmax(dim=1)
            pred_cpu = torch.cat(predictions).cpu()
            labels_cpu =  torch.cat(all_labels).cpu()
            torch.save(
                {   'predict_proba' : m(torch.cat(predict_proba)),
                    'predictions' : pred_cpu.numpy(),
                    'labels_val' : labels_cpu.numpy(),
                    'model_state_dict' : model.state_dict(),
                    'f1_score' : f1_score(pred_cpu.numpy(),labels_cpu.numpy(), average = 'macro'),
                    'accuracy' : accuracy_score(pred_cpu.numpy(),labels_cpu.numpy())
            },  '/home/jovyan/Tomics-CP-Chem-MoA/saved_models/' + model_name + '.pt'
            )
    else:
        ValueError("No validation metric identified")
    return val_metric

''''
def best_val_metric(val_str):
    
    The initial validation metric number to compare and see if performance has improved
    Input:
        val_str: string, fe, f1, or loss
    Output:
        val_metric: float, 0 or np.inf depending on the validation metric
    if val_str == "fe":
        return 0
    elif val_str == "f1":
        return 0
    elif val_str == "loss":
        return np.inf
    else:
        ValueError("No validation metric identified")
'''
# ---------------------------------- Checking that the data is correct ---------------------------------- #

def cmpd_id_overlap_check(train_data_lst, valid_data_lst, test_data_lst):
    # check to make sure the compound IDs do not overlapp
    inter1 = set(test_data_lst) & set(train_data_lst)
    inter2 = set(test_data_lst) & set(valid_data_lst)
    inter3 = set(train_data_lst) & set(valid_data_lst)
    assert len(inter1) + len(inter2) + len(inter3) == 0, ("There are overlapping compounds between the training, validation and test sets")

def inputs_equalto_labels_check(training_df, train_labels, validation_df, validation_labels, test_df, test_labels):
    # make sure that the number of labels is equal to the number of inputs
    assert len(training_df) == len(train_labels)
    assert len(validation_df) == len(validation_labels)
    assert len(test_df) == len(test_labels)

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
        assert all_profiles.shape[0] == 13660, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == 121, 'Incorrect number of unique compounds'
    elif file == 'erik10':
        assert all_profiles.shape[0] == 18042, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == int(243 -2), 'Incorrect number of unique compounds'
    elif file == 'erik10_hq':
        assert all_profiles.shape[0] == 4564, 'Incorrect number of profiles'
        assert all_profiles["Compound ID"].nunique() == 185, 'Incorrect number of unique compounds'
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

def check_overlap_sigid(L1000_training, L1000_validation, L1000_test):
    # check to make sure the sig_ids do not overlap
    inter1 = set(list(L1000_training["sig_id"])) & set(list(L1000_validation["sig_id"]))
    inter2 = set(list(L1000_training["sig_id"])) & set(list(L1000_test["sig_id"]))
    inter3 = set(list(L1000_validation["sig_id"])) & set(list(L1000_test["sig_id"]))
    assert len(inter1) + len(inter2) + len(inter3) == 0, ("There is an intersection between the training, validation and test sets")


def getting_correct_image_indices(method, file_name, df_total):
    '''
    For IGTD and DWTM, I need to get the correct indices for the images based on the index of fold 0. This function:
    1. Gets the correct indices for the images for folds 1 - 4 by comparing the values in the dataframe to the values in the dataframe of fold 0
    2. Checks that there is no overlap of within folds or across the test set folds
    3. Places the indices in a dictionary and saves it as a pickle
    Input:
        method: string, either "IGTD" or "DWTM"
        file_name: string, name of the data set
        df_total: pandas dataframe, dataframe of the entire concateated data set representing fold 0.
    Output:
        Saves a dictionary with indices for each fold, each index corresponding to the correct image
    '''
    def checking_index_overlap_within_fold(fold_dict_index):
        # checks that there is no overlap of indices within a fold
        inter1 = set(fold_dict_index["train"]) & set(fold_dict_index["valid"])
        inter2 = set(fold_dict_index["train"]) & set(fold_dict_index["test"])
        inter3 = set(fold_dict_index["valid"]) & set(fold_dict_index["test"])
        assert len(inter1) + len(inter2) + len(inter3) == 0, ("There is an intersection between the training, validation and test sets")
    
    def checking_index_overlap_between_test(fold_index_dictionaries):
        # checks that there is no overlap of indices between the test folds
        inter1 = set(fold_index_dictionaries[0]["test"]) & set(fold_index_dictionaries[1]["test"])
        inter2 = set(fold_index_dictionaries[0]["test"]) & set(fold_index_dictionaries[2]["test"])
        inter3 = set(fold_index_dictionaries[0]["test"]) & set(fold_index_dictionaries[3]["test"])
        inter4 = set(fold_index_dictionaries[1]["test"]) & set(fold_index_dictionaries[2]["test"])
        inter5 = set(fold_index_dictionaries[1]["test"]) & set(fold_index_dictionaries[3]["test"]) 
        inter6 = set(fold_index_dictionaries[2]["test"]) & set(fold_index_dictionaries[3]["test"])
        inter7 = set(fold_index_dictionaries[4]["test"]) & set(fold_index_dictionaries[3]["test"])
        inter8 = set(fold_index_dictionaries[1]["test"]) & set(fold_index_dictionaries[4]["test"]) 
        inter9 = set(fold_index_dictionaries[2]["test"]) & set(fold_index_dictionaries[4]["test"])
        inter10 = set(fold_index_dictionaries[0]["test"]) & set(fold_index_dictionaries[4]["test"])
        return len(inter1) + len(inter2) + len(inter3) + len(inter4) + len(inter5) + len(inter6) + len(inter7) + len(inter8) + len(inter9) + len(inter10) == 0, ("There is an intersection between the test sets")
    
    # clue row metadata with rows representing transcription levels of specific genes
    clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")
    fold_index_dictionaries = []
    for fold_int in range(0,5):
        print(f'Fold Iteration: {fold_int}')
        training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
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
        for df_split, df_split_str in [(df_train_features, "train"), (df_val_features, "valid"), (df_test_features, "test")]:
            print(f'Fold Iteration: {fold_int}, Split: {df_split_str}')
            if method == 'DWTM':
                df_original = df_total
                df_original = df_original.drop("Class", axis = 1)
            elif method == 'IGTD':
                df_original = df_total
            else:
                ValueError('Method must be either DWTM or IGTD')
                    # reseting index of both dataframes

            df_original = df_original.reset_index().rename(columns={'index': 'original_index'})
            # Create a dictionary to map the current column names to their string representations
            column_mapping = {col: str(col) for col in df_split.columns}
            # Update the column names in the dataframe using the rename() function
            df_split = df_split.rename(columns=column_mapping)
            df_original = df_original.rename(columns=column_mapping)
            df_split.reset_index().rename(columns={'index': 'split_index'})
            # Get the common columns to merge on, excluding the 'index' column
            common_columns1 = [col for col in df_original.columns if col != 'original_index']
            common_columns2 = [col for col in df_split.columns if col != 'index']
            assert common_columns1 == common_columns2, "The columns in the original and split dataframes should match"
            # Merge the dataframes on all common columns
            merged_df = df_original.merge(df_split, on= ['0', '1'] )
            assert merged_df.shape[0] == df_split.shape[0], "The number of rows in the merged dataframe should match the number of rows in the split dataframe"
            # Extract the index from the merged dataframe
            matching_indices = merged_df['original_index'].values

            fold_dict_index[df_split_str] = matching_indices
        checking_index_overlap_within_fold(fold_dict_index)
        fold_index_dictionaries.append(fold_dict_index)
    checking_index_overlap_between_test(fold_index_dictionaries)
    with open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/" + method + '_' + file_name + "_splits.pkl", 'wb') as f:
        pickle.dump(fold_index_dictionaries, f)

def extracting_pretrained_single_models(single_model_str, fold_num):
    if fold_num == 0:
        if single_model_str == "GE":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/GE_fold0_model_checkpoint_559.pt'
        elif single_model_str == "CP":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CP_fold0_model_checkpoint_606.pt'
        elif single_model_str == "CS":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CS_fold0_model_checkpoint_584.pt'
        else:
            ValueError("single_model_str must be either GE, CP, or CS")
    elif fold_num == 1:
        if single_model_str == "GE":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/GE_fold1_model_checkpoint_560.pt'
        elif single_model_str == "CP":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CP_fold1_model_checkpoint_610.pt'
        elif single_model_str == "CS":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CS_fold1_model_checkpoint_585.pt'
        else:
            ValueError("single_model_str must be either GE, CP, or CS")
    elif fold_num == 2:
        if single_model_str == "GE":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/GE_fold2_model_checkpoint_561.pt'
        elif single_model_str == "CP":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CP_fold2_model_checkpoint_611.pt'
        elif single_model_str == "CS":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CS_fold2_model_checkpoint_586.pt'
        else:
            ValueError("single_model_str must be either GE, CP, or CS")
    elif fold_num == 3:
        if single_model_str == "GE":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/GE_fold3_model_checkpoint_562.pt'
        elif single_model_str == "CP":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CP_fold3_model_checkpoint_612.pt'
        elif single_model_str == "CS":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CS_fold3_model_checkpoint_587.pt'
        else:
            ValueError("single_model_str must be either GE, CP, or CS")
    elif fold_num == 4:
        if single_model_str == "GE":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/GE_fold4_model_checkpoint_563.pt'
        elif single_model_str == "CP":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CP_fold4_model_checkpoint_613.pt'
        elif single_model_str == "CS":
            return '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/Final_Pretrained_Single_Models/CS_fold4_model_checkpoint_588.pt'
        else:
            ValueError("single_model_str must be either GE, CP, or CS")
    else:
        ValueError("fold_num must be between 0 and 4")

def optuna_combinations_feature_extraction(trial, num_feat, num_classes, training_generator, validation_generator, pretrained_model, model_name, device):
    class Extended_Model(nn.Module):
        def __init__(self, trial, num_features, pretrained_model, num_classes):
            super(Extended_Model, self).__init__()
            self.base_model = pretrained_model
            self.num_features = num_features
            self.num_classes = num_classes
            n_layers = trial.suggest_int('n_layers', 1, 4)
            layers = []
            for i in range(n_layers):
                out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 250)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.LeakyReLU())
                layers.append(nn.BatchNorm1d(out_features))
                p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5)
                layers.append(nn.Dropout(p))
                in_features = out_features
            layers.append(nn.Linear(out_features, num_classes))

            # Additional layers for feature extraction
            self.additional_layers = nn.Sequential(*layers)

        def forward(self, x):
            x = self.base_model(x)
            x = self.additional_layers(x)
            return x
    

    model = Extended_Model(trial, num_feat, pretrained_model, num_classes)
# generate the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))


    yn_class_weights = trial.suggest_categorical('yn_class_weights', [True, False])
    if yn_class_weights:     # if we want to apply class weights
        if model_name == 'CP': # we want CP first if possible, since that is our driver
            class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
        elif model_name == 'GE': # otherwise GE
            class_weights = apply_class_weights_CP(train_np, L1000_training, dict_moa, device)
        class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'focal', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false','ols'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        

    max_epochs = 150

    
#------------------------------   Calling functions --------------------------- #
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = scheduler,
                model_name=model_name,
                device = device,
                val_str = 'fe',
                early_patience = 0)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return val_f1_score_per_epoch[-1]

def optuna_combinations_feature_tuning(trial, num_classes, training_generator, validation_generator, model, model_name, device):
    # generate the model
# generate the optimizer
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "CosineAnnealingLR"])
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=trial.suggest_int("step_size", 5, 30), gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=trial.suggest_float("gamma", 0.1, 0.9))
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trial.suggest_int("T_max", 5, 30))


    yn_class_weights = trial.suggest_categorical('yn_class_weights', [True, False])
    if yn_class_weights:     # if we want to apply class weights
        class_weights = apply_class_weights_GE(train_np, L1000_training, dict_moa, device)
    else:
        class_weights = None
    loss_fn_str = trial.suggest_categorical('loss_fn', ['cross', 'focal', 'BCE'])
    loss_fn_train_str = trial.suggest_categorical('loss_train_fn', ['false','ols'])
    loss_fn_train, loss_fn = different_loss_functions(
                                                      loss_fn_str= loss_fn_str,
                                                      loss_fn_train_str = loss_fn_train_str,
                                                      class_weights = class_weights)

    
    if loss_fn_train_str == 'ols':
        from ols import OnlineLabelSmoothing
        loss_fn_train = OnlineLabelSmoothing(alpha = trial.suggest_float('alpha', 0.1, 0.9),
                                          n_classes=num_classes, 
                                          smoothing = trial.suggest_float('smoothing', 0.001, 0.3))
        

    max_epochs = 150

    
#------------------------------   Calling functions --------------------------- #
    train_loss_per_epoch, train_acc_per_epoch, val_loss_per_epoch, val_acc_per_epoch, val_f1_score_per_epoch, num_epochs = adapt_training_loop(n_epochs = max_epochs,
                optimizer = optimizer,
                model = model,
                loss_fn = loss_fn,
                loss_fn_train = loss_fn_train,
                loss_fn_str = loss_fn_str,
                train_loader=training_generator, 
                valid_loader=validation_generator,
                my_lr_scheduler = scheduler,
                model_name=model_name,
                device = device,
                val_str = 'f1',
                early_patience = 15)


    #lowest1, lowest2 = find_two_lowest_numbers(val_loss_per_epoch)
    return max(val_f1_score_per_epoch)

def CP_driving_code(file_name, fold_int):
    # clue row metadata with rows representing transcription levels of specific genes
    clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")



    training_set, validation_set, test_set =  accessing_all_folds_csv(file_name, fold_int)
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    variance_thresh = 0
    normalize_c = 'False'

    npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c)

    df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
            clue_gene, 
            npy_exists = npy_exists,
            use_variance_threshold = variance_thresh, 
            normalize = normalize_c, 
            save_npy = save_npy,
            data_subset = file_name)
    checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)


    # ----------------------------------------- Prepping Chemical Structure Data ---------------------------------------#
    # download dictionary which associates moa with a tensor

    dict_moa = dict_splitting_into_tensor(training_set)
    assert set(training_set.moa.unique()) == set(validation_set.moa.unique()) == set(test_set.moa.unique()), "Training, validation and test sets have different labels"
    assert df_train_features.shape[0] == df_train_features.dropna().shape[0], "NaNs in training set"


    num_classes = len(training_set.moa.unique())


    # split data into labels and inputs
    training_df, train_labels = splitting(training_set)
    validation_df, validation_labels = splitting(validation_set)
    test_df, test_labels = splitting(test_set)



    # make sure that the number of labels is equal to the number of inputs
    inputs_equalto_labels_check(training_df, train_labels, validation_df, validation_labels, test_df, test_labels)



    num_classes = len(np.unique(df_train_labels["moa"]))
    
    # extract compound IDs
    test_data_lst= list(test_set["Compound_ID"].unique())
    train_data_lst= list(training_set["Compound_ID"].unique())
    valid_data_lst= list(validation_set["Compound_ID"].unique())

    cmpd_id_overlap_check(train_data_lst, valid_data_lst, test_data_lst)

    # downloading paths to all of the images
    paths_v1v2 = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_treated_v1v2.csv')

    # 
    training_df = paths_v1v2[paths_v1v2["compound"].isin(train_data_lst)].reset_index(drop=True)
    validation_df = paths_v1v2[paths_v1v2["compound"].isin(valid_data_lst)].reset_index(drop=True)
    test_df = paths_v1v2[paths_v1v2["compound"].isin(test_data_lst)].reset_index(drop=True)



    # importing data normalization pandas dataframe
    pd_image_norm = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')
    return training_df, validation_df, test_df, paths_v1v2, df_train_labels, dict_moa, pd_image_norm, df_val_labels, df_test_labels, num_classes, training_set, validation_set, test_set

def GE_driving_code(file_name, fold_int):
    # clue row metadata with rows representing transcription levels of specific genes
    clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")


    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
    #file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    # Creating a  dictionary of the one hot encoded labels
    dict_moa = dict_splitting_into_tensor(training_set)


    # checking that no overlap in sig_id exists between training, test, validation sets
    check_overlap_sigid(L1000_training, L1000_validation, L1000_test)
    # shuffling training and validation data
    L1000_training = L1000_training.sample(frac = 1, random_state = 1)
    L1000_validation = L1000_validation.sample(frac = 1, random_state = 1)
    L1000_test = L1000_test.sample(frac = 1, random_state = 1)

    print("extracting training transcriptomes")
    profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
    train_np = np_array_transform(profiles_gc_too_train)
    print("extracting validation transcriptomes")
    profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
    valid_np = np_array_transform(profiles_gc_too_valid)
    print("extracting test transcriptomes")
    profiles_gc_too_test = tprofiles_gc_too_func(L1000_test, clue_gene)
    test_np = np_array_transform(profiles_gc_too_test)


    num_classes = len(L1000_training["moa"].unique())


    dict_cell_lines = extract_all_cell_lines(pd.concat([L1000_training, L1000_validation, L1000_test]))
    num_cell_lines = len(dict_cell_lines)
    return train_np, valid_np, test_np, num_classes, dict_cell_lines, num_cell_lines, dict_moa, training_set, validation_set, test_set, L1000_training, L1000_validation, L1000_test

def returning_smile_string(compound_df, CID):
    '''
    Necessary to return the smile string a vector and not as a pandas data frame.
    This is to handle enantiomers that are present, though not many examples exist in the dataset.
    '''
    smile_string = compound_df["SMILES"][compound_df["Compound_ID"]== CID] 
    if smile_string.shape[0] > 1:
        smile_string= smile_string.sample(n=1, random_state=1)
        smile_string = smile_string.iloc[0]
    if type(smile_string) == pd.Series:
        smile_string = smile_string.values[0]
    elif type(smile_string) == str:
        smile_string = smile_string
    else:
        print("We have a problem")
    return smile_string

def dubbelcheck_dataset_length(split, driver, fold_int):
    """
    This functions is used to check the length of the dataset for each fold and driver.
    """
    if split == "train":
        if driver == "CP":
            if fold_int == 0:
                num_examples = 5337
            elif fold_int == 1:
                num_examples = 5454
            elif fold_int == 2:
                num_examples = 5400
            elif fold_int == 3:
                num_examples = 5499
            elif fold_int == 4:
                num_examples = 5499
            else:
                ValueError("Fold not recognized")
            
        elif driver == "GE":
            if fold_int == 0:
                num_examples = 1147
            elif fold_int == 1:
                num_examples = 1082
            elif fold_int == 2:
                num_examples = 1654
            elif fold_int == 3:
                num_examples = 1487
            elif fold_int == 4:
                num_examples = 1885
            else:
                ValueError("Fold not recognized")

        else:
            ValueError("Driver not recognized")
    elif split == "valid":
        if driver == "CP":
            if fold_int == 0:
                num_examples = 2376
            elif fold_int == 1:
                num_examples = 2313
            elif fold_int == 2:
                num_examples = 2322
            elif fold_int == 3:
                num_examples = 2322
            elif fold_int == 4:
                num_examples = 2322
            else:
                ValueError("Fold not recognized")
            
            
        elif driver == "GE":
            if fold_int == 0:
                num_examples = 810
            elif fold_int == 1:
                num_examples = 509
            elif fold_int == 2:
                num_examples = 328
            elif fold_int == 3:
                num_examples = 407
            elif fold_int == 4:
                num_examples = 239
            else:
                ValueError("Fold not recognized")
            
        else:
            ValueError("Driver not recognized")
    elif split == "test":
        if driver == "CP":
            if fold_int == 0:
                num_examples = 1998 
            elif fold_int == 1:
                num_examples = 1994
            elif fold_int == 2:
                num_examples = 1989
            elif fold_int == 3:
                num_examples = 1890
            elif fold_int == 4:
                num_examples = 1890
            else:
                ValueError("Fold not recognized")
            
        elif driver == "GE":
            if fold_int == 0:
                num_examples = 430
            elif fold_int == 1:
                num_examples = 796
            elif fold_int == 2:
                num_examples = 405
            elif fold_int == 3:
                num_examples = 493
            elif fold_int == 4:
                num_examples = 263
            else:
                ValueError("Fold not recognized")
            
            ValueError("Driver not recognized")
    else:
        ValueError("Split not recognized")
    return num_examples

'''
def choose_cell_lines_to_include(df_set, clue_sig_in_SPECS, cell_lines):
    Returns training/validation/test set with only the cell lines specified by the user
    Input:
        df_set: a dataframe with the training/validation/test data
        cell_lines: a dictionary, where the key is the name of the moa and value is a list with the names of cell lines to be included.
            Default is empty. Ex. "{"cyclooxygenase inhibitor": ["A375", "HA1E"], "adrenergic receptor antagonist" : ["A375", "HA1E"] }"
    Output:
        df_set: a dataframe with the training/validation/test data with only the cell lines specified by the user
    profile_ids = clue_sig_in_SPECS[["Compound ID", "sig_id", "moa", 'cell_iname']][clue_sig_in_SPECS["Compound ID"].isin(df_set["Compound_ID"].unique())]
    return profile_ids[profile_ids["cell_iname"].isin(cell_lines)]
'''
'''

def accessing_correct_fold_csv_files(file):
    
  #  This function returns the correct, fold-0 train, valid and test split data csvs given a file name
   # Input:
  #      file: the name of the file (type = string)
   # Output:
  #      training_set: the training set (type = pandas dataframe)
 #       validation_set: the validation set (type = pandas dataframe)
  #      test_set: the test set (type = pandas dataframe)

 # download csvs with all the data pre split''
    if file == 'tian10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/tian10/'
        train_filename = 'tian10_clue_train_fold_0.csv'
        val_filename = 'tian10_clue_val_fold_0.csv'
        test_filename = 'tian10_clue_test_fold_0.csv'
    elif file == 'tian10_all':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/tian10/'
        train_filename = 'tian10_all_train_fold_1.csv'
        val_filename = 'tian10_all_val_fold_1.csv'
        test_filename = 'tian10_all_test_fold_1.csv'
    elif file == 'erik10':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_train_fold_0.csv'
        val_filename = 'erik10_clue_val_fold_0.csv'
        test_filename = 'erik10_clue_test_fold_0.csv'
    elif file == 'erik10_all':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_all_train_fold_1.csv'
        val_filename = 'erik10_all_val_fold_1.csv'
        test_filename = 'erik10_all_test_fold_1.csv'
    elif file == 'erik10_hq':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_hq_train_fold_0.csv'
        val_filename = 'erik10_clue_hq_val_fold_0.csv'
        test_filename = 'erik10_clue_hq_test_fold_0.csv'
    elif file == 'erik10_8_12':
        dir_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
        train_filename = 'erik10_clue_8_12__train_fold_1.csv'
        val_filename = 'erik10_clue_8_12__val_fold_1.csv'
        test_filename = 'erik10_clue_8_12__test_fold_1.csv'
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
'''

# ---------------------------------------------- data loading ----------------------------------------------#
# ----------------------------------------------------------------------------------------------------------#
'''
def load_train_valid_data(path, train_data, valid_data, test_data = None):
    
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
    
    if test_data:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation = pd.read_csv(path + valid_data, delimiter = ",")
        L1000_test = pd.read_csv(path + test_data, delimiter = ",")
        return L1000_training, L1000_validation, L1000_test
    else:
        L1000_training = pd.read_csv(path + train_data, delimiter = ",")
        L1000_validation = pd.read_csv(path + valid_data, delimiter = ",")
    return L1000_training, L1000_validation
'''
 
# ---------------------------------------------- data preprocessing ----------------------------------------------#
'''
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
        x_train_var = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val_var = x_val.loc[:,var_thresh.variances_ > var_threshold]
        if x_train_var.shape[1] < x_train.shape[1]:
            print("Number of features removed due to Variance Threshold: ", x_train.shape[1] - x_train_var.shape[1])
        return x_train_var, x_val_var
    else:
        x_test_var = x_test.loc[:,var_thresh.variances_ > var_threshold]
        x_train_var = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
        x_val_var = x_val.loc[:,var_thresh.variances_ > var_threshold]
        if x_train_var.shape[1] < x_train.shape[1]:
            print("Number of features removed due to Variance Threshold: ", x_train.shape[1] - x_train_var.shape[1])

        return x_train_var, x_val_var, x_test_var
    
'''