c#!/usr/bin/env python
# coding: utf-8

# In[90]:


#!/usr/bin/env python
# coding: utf-8

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
import math 

import umap
import math


# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn


from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve,log_loss, accuracy_score, f1_score
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.ensemble import VotingClassifier
import os
import time
from time import time
import datetime
import pandas as pd
import numpy as np
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from skmultilearn.adapt import MLkNN

# CMAP (extracting relevant transcriptomic profiles)
from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
import time
import joblib

from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.metrics import precision_recall_curve,log_loss
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.feature_selection import VarianceThreshold
import os
import pandas as pd
import numpy as np
import torch
import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
nn._estimator_type = "classifier"

# -------------------------------------------------------------------------------------------------------------------------# 
# Prepping Neptune.ai for logging --------------------------------------------------------------------------------------------
import neptune.new as neptune


# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# -------------------------------------------------------------------------------------------------------------------------
def load_train_valid_data(train_data, valid_data):
    '''
    Functions loads the data frames that will be used to train classifier and assess its accuracy in predicting.
    input:
        train_data: filename of training csv file
        valid_data: filename of validation csv file
    ouput:
       L1000 training: pandas dataframe with training data
       L1000 validation: pandas dataframe with validation data
    '''
    path = '/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/data_split_csvs/'
    L1000_training = pd.read_csv(path + train_data, delimiter = ",")
    L1000_validation =pd.read_csv(path + valid_data, delimiter = ",")
    return L1000_training, L1000_validation

def variance_threshold(x_train, x_val):
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
    var_thresh = VarianceThreshold(threshold = 0.8) # sets a variance threshold
    var_thresh.fit(x_train) # learn empirical variances from X
    x_train = x_train.loc[:,var_thresh.variances_ > 0.8] # locate all variance thresholds above 0.8, keep those columns
    x_val = x_val.loc[:,var_thresh.variances_ > 0.8]
    return x_train, x_val

def normalize_func(trn, test):
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
    # norm_model = StandardScaler()
    trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
    tst_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
    return trn_norm, tst_norm


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
    '''returns transcriptomic profile of of specific ID with in the form of a numpy array
    
    input:
     profiles_gc_too: gc_too dataframe hosting transcriptomic profiles
     idx:  extract unique column name from L1000 data
    
    output: 
      numpy array of a single transcriptomic profile
    '''
    tprofile_id =  profiles_gc_too.col_metadata_df.iloc[idx]
    tprofile_id_sig = [tprofile_id.name] 
    tprofile_gctoo = sg.subset_gctoo(profiles_gc_too, cid= tprofile_id_sig) 
    #return torch.tensor(tprofile_gctoo.data_df.values.astype(np.float32)) 
    return tprofile_id_sig, np.asarray(tprofile_gctoo.data_df.values.astype(np.float32))    


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

def acquire_npy(dataset):
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
        npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/data_splits_npy2_moas_train.npy')
    elif dataset == 'val':
        #filename = input('Give name of npy file (str): ')
        #npy_set = np.load(path + filename)
        npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/data_splits_npy2_moas_valid.npy')
    else:
        filename =  input('Give name of npy file (str): ')
        npy_set = np.load(path + filename)
    return pd.DataFrame(npy_set)

def save_npy(dataset):
    '''Save the numpy array of the selected transcriptomics profiles
    Input:
        dataset: the numpy array to be saved
    '''
    path = '/scratch2-shared/erikep/data_splits_npy/'
    file_name = input("Give filename for numpy array: ")
    np.save(path + file_name, dataset)

def get_models(class_weight):
    '''
    Input:
        class weight: including or not including class weight.
    Output:
        A list of tuples, with a str with a descriptor followed by the classifier function)
    '''
    TNC = TabNetClassifier()
    TNC._estimator_type = "classifier"
    models = list()
    models.append(('logreg', LogisticRegression(class_weight = class_weight, solver= "liblinear", penalty = "l2"))) 
    #models.append(('RFC',RandomForestClassifier(class_weight= class_weight))) 
    #models.append(('gradboost', GradientBoostingClassifier()))
    #models.append(('Ada', AdaBoostClassifier()))
    #models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
    #models.append(('Bagg',BaggingClassifier()))
    #models.append(('Tab', TNC))
    return models

def printing_results(class_alg, labels_val, predictions): 
    '''
    Printing the results from the 
    Input:
        class_alg: name of the model
        labels_val: the correct guesses
        predictions: the predictions made by the model
    Output:
        Printed results of accuracy, F1 score, and confusion matrix
    '''
    print('----------------------------------------------------------------------')
    print(class_alg)
    anders = f1_score(labels_val, predictions, labels = ["0","1"], average = 'macro')
    print(f' Accuracy score: {accuracy_score(labels_val, predictions)}')
    print(f' F1 Score: {     anders   }')
    print(f' Confusion Matrix: {confusion_matrix(labels_val, predictions)}')
    print('----------------------------------------------------------------------')

def write_list(a_list, file_type):
    with open('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/pickles/tabnet_pickles/' + file_type + '.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
    print('Done writing binary file')

def save_val(a_list, file_type):
    with open('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/pickles/val_order_pickles/' + file_type + '.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
    print('Done writing binary file')

# ------------------------------------------------------------------------------------------------------------------------------
def main(use_variance_threshold, normalize, L1000_training, L1000_validation, clue_gene, npy_exists, apply_class_weight, ensemble):
    '''
    Tests a series of ML algorithms after optional pre-processing of the data in order to make predictions on the MoA class based on
    chosen transcriptomic profiles. 

    Input:
        use_variance_threshold: True/False (also have to adjust hyperparameter in the function itself depending on normalization.)
        normalize: True/False. Whether or not to normalize the data.
        L1000_training: Str. Name of the csv file with training rows
        L1000_validation: Str. Name of the csv file with validation rows
        clue_gene: Row metadata fro the transcriptomic profiles
        npy_exists: True/False: whether or not the numpy array with transcriptomic profiles has already been created (can save time if many moas are used.)
        apply_class_weight: True/False. Whether to apply class weights for the random forest classifier.
        ensemble: True/False. Whether to apply to do an ensemble classifier with a select number of classifiers.
    Output:
        Prints the accuracy, F1 score and confusion matrix for each of the ML algorithms.
        Save unique numpy array. 
    '''
    # shuffling training and validation data
    L1000_training = L1000_training.sample(frac = 1, random_state = 1)
    L1000_validation = L1000_validation.sample(frac = 1, random_state = 1)
    # extracting training transcriptomes
    profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
    if npy_exists:
        df_train = acquire_npy('train')
    else:    
        df_train = np_array_transform(profiles_gc_too_train)
        #save_npy(df_train)
    
    #input_df_train, labels_train = splitting(L1000_training) 
    #print(sig_id_df_check.head(20))
    #print(input_df_train.sig_id.head(20))
    #assert sig_id_df_check.all() == input_df_train.sig_id.all()
    
    # create dictionary where moas are associated with a number

    '''
    # extracting valid 
    profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
    if npy_exists:
        df_val = acquire_npy('val')
    else:    
        df_val = np_array_transform(profiles_gc_too_valid)
        save_npy(df_val)
    input_df_val, labels_val = splitting(L1000_validation) 

    
    # to normalize
    if normalize:
        df_train, df_val = normalize_func(df_train, df_val)
    
    # applying class weights
    if apply_class_weight:
        class_weight = "balanced"
    else:
        class_weight = None
    
    models = get_models(class_weight)
    scores = list()
    # battery of classifiers
    for class_alg in models:
        classifier = class_alg[1]
        # use variance threshold
        if use_variance_threshold:
            df_train_vs, df_val_vs = variance_threshold(df_train, df_val)
            classifier.fit(df_train_vs.values, labels_train.values)
            predictions = classifier.predict(df_val_vs.values)
            if class_alg[0] == 'Tab':
                save_val(labels_val, 'tab_val')
                class_probs= classifier.predict_proba(df_val_vs.values)
                write_list(predictions, 'predictions')
                write_list(class_probs, 'class_probs')

        else:
            classifier.fit(df_train.values, labels_train.values)
            predictions = classifier.predict(df_val.values)
            if class_alg[0] == 'Tab':
                save_val(labels_val, 'tab_val')
                class_probs = classifier.predict_proba(df_val_vs.values)
                write_list(predictions, 'predictions')
                write_list(class_probs, 'class_probs')
        f1_score_from_model = f1_score(labels_val, predictions, average= "macro") 
        scores.append(f1_score_from_model)
        printing_results(class_alg, labels_val, predictions)
       


    if ensemble:
        # 'soft':  predict the class labels based on the predicted probabilities p for classifier 
        ensemble = VotingClassifier(estimators = models, voting = 'soft', weights = scores)
        
        if use_variance_threshold:
                df_train_vs, df_val_vs = variance_threshold(df_train, df_val)
                ensemble.fit(df_train_vs.values, labels_train.values)
                predictions = ensemble.predict(df_val_vs.values)

        else:
            ensemble.fit(df_train.values, labels_train.values)
            predictions = ensemble.predict(df_val.values)
        
        printing_results('ensemble', labels_val, predictions)'''
    return df_train


# In[91]:


if __name__ == "__main__":  
    # train_filename = input('Training Data Set Filename: ')
    #valid_filename = input('Validation Data Set Filename: ')
    #train_filename = 'L1000_training_set_cyclo_adr_2.csv'
    #valid_filename = 'L1000_test_set_cyclo_adr_2.csv'
    train_filename = "L1000_training_set_nv_cyc_adr.csv"
    valid_filename = "L1000_test_set_nv_cyc_adr.csv" 
    #train_filename = 'L1000_training_set.csv'
    # valid_filename = 'L1000_valid_set.csv'
    
    # loading data
    L1000_training, L1000_validation =  load_train_valid_data(train_filename, valid_filename)
    df_train = main(use_variance_threshold = False, 
         normalize= True, 
         L1000_training = L1000_training, 
         L1000_validation = L1000_validation, 
         clue_gene= clue_gene, 
         npy_exists = False,
         apply_class_weight= True,
         ensemble = False)



moa_dictionary = {}
for i,j in enumerate(L1000_training.moa.unique()):
    moa_dictionary[j] = i


run = neptune.init_run(project='erik-everett-palm/Tomics-PCA-UMAP')
run["parameters/moa_dictionary"] = str(moa_dictionary)
run["parameters/train_filename"] = train_filename
norm = False
if norm:
    scaler = StandardScaler()
    df_train[df_train.columns[:-1]] = StandardScaler().fit_transform(df_train[df_train.columns[:-1]])
    run["parameters/normalize"] = str(scaler)
else:
    run["parameters/normalize"] = "None"
    
df_train = pd.merge(df_train, L1000_training[["sig_id", "moa"]], how = "outer", on ="sig_id")




df_train.drop(columns = ["sig_id"], inplace = True)


print("starting PCA")
pca_ten = PCA(n_components=10)


df_train_features = df_train[df_train.columns[:-1]]

pca_ten.fit_transform(df_train_features)


pca_comp = plt.figure()
plt.bar([i for i in range(0,10)], pca_ten.explained_variance_ratio_)
plt.title("Explained Variance Ratio of PCA Components")

run["images/pca_components"] = neptune.types.File.as_image(pca_comp)



pca = PCA(n_components=2)



principalComponents = pca.fit_transform(df_train_features)
principalDf = pd.DataFrame(data = principalComponents
             ,#  columns = [f' PC1: VarExp: {pca.explained_variance_[0]}', f' PC1: VarExp: {pca.explained_variance_[1]}'])
              columns = ["PC1" , "PC2"])

principalDf["moa"] =  df_train["moa"]





pca_plot = plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Dataset",fontsize=20)
targets = [i for i in moa_dictionary.keys()]
# 
#
colors = [sns.color_palette()[x] for x in moa_dictionary.values()]
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['moa'] == target
    plt.scatter(principalDf.loc[indicesToKeep, "PC1"], 
                principalDf.loc[indicesToKeep, 'PC2'],  
                c= color,
                s = 10)

plt.legend(targets,prop={'size': 15})
run["images/pca_plot"] = neptune.types.File.as_image(pca_plot)



pca = PCA(n_components=20)
principalComponents = pca.fit_transform(df_train_features)
principalDf = pd.DataFrame(data = principalComponents
                           )

pca.explained_variance_ratio_
principalDf

principalDf["moa"] =  df_train["moa"]






#reducer = umap.UMAP(n_components = 2, min_dist = 1, n_neighbors = math.sqrt(df_train.shape[0]), init = principalDf, n_epochs = 1000)

print("Starting UMAP")
# In[127]:
umap_neighbors = 15
umap_min_dist = 0.1
run["parameters/umap_neighbors"] = umap_neighbors
run["parameters/umm_min_dist"] = umap_min_dist

reducer = umap.UMAP(n_neighbors= umap_neighbors, min_dist = umap_min_dist)


# In[128]:


# change moa to classes using the above dictionary
for i in range(principalDf.shape[0]):
    principalDf.iloc[i, -1] = moa_dictionary[principalDf.iloc[i, -1]]



embedding = reducer.fit_transform(principalDf)


# In[132]:


'''plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
     c=
     [sns.color_palette()[x] for x in principalDf.moa.map({0:0, 1:1 })])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24);
plt.show()
'''


# In[142]:


targets = [i for i in moa_dictionary.keys()]
plt.figure()
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in principalDf.moa], 
    s = 5)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Penguin dataset', fontsize=24);
plt.legend(targets , prop={'size': 7})



# In[162]:


import umap.plot
plt.figure(figsize=(10,10))
umap.plot.points(reducer, labels= principalDf.moa, theme = "fire")
plt.title('UMAP projection of the dataset', fontsize=24)
umap.plot.plt.savefig("umap.png")

import matplotlib.image as mpimg
umap_img = mpimg.imread('umap.png')
run["images/umap_plot"] = neptune.types.File.as_image(umap_img)

run.stop()



