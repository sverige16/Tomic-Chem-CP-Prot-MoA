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
import neptune.new as neptune
import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import pre_processing, create_splits, load_train_valid_data, dict_splitting_into_tensor

def cmpd_clustering(df_train_features, df_train_labels, moa_subset):
    # Subset the data to only include the MoA of interest
    df_singleMoA_labels = df_train_labels[df_train_labels["moa"] == moa_subset]
    df_singleMoA_train_features = df_train_features.iloc[df_singleMoA_labels.index]
    df_singleMoA_train_features.reset_index(drop=True, inplace=True)
    df_singleMoA_labels.reset_index(drop=True, inplace=True)
    
    print("Investigating using PCA")
    pca_ten = PCA(n_components=10)
    pca_ten.fit_transform(df_singleMoA_train_features)
    pca_comp = plt.figure()
    plt.bar([i for i in range(0,10)], pca_ten.explained_variance_ratio_)
    plt.title("Explained Variance Ratio of PCA Components")
    run["images/pca_compoenents"].upload(pca_comp)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_singleMoA_train_features)
    principalDf = pd.DataFrame(data = principalComponents
                ,#  columns = [f' PC1: VarExp: {pca.explained_variance_[0]}', f' PC1: VarExp: {pca.explained_variance_[1]}'])
                columns = ["PC1" , "PC2"])

    pca_plot = plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel(f' Principal Component - 1: {round(pca.explained_variance_[0], 2)}', fontsize=20)
    plt.ylabel(f' Principal Component - 2: {round(pca.explained_variance_[1], 2)}', fontsize=20)
    plt.title(f' Principal Component Analysis of {moa_subset} Dataset', fontsize=20)
    targets =  [i for i in df_singleMoA_labels["Compound ID"].unique()]
    labels_to_targets = {label: i for i, label in enumerate(targets)}
    colors = [labels_to_targets[label] for label in df_singleMoA_labels["Compound ID"]]
    pca_plot = plt.figure(figsize=(15,10))
    graph = plt.scatter(
        principalDf["PC1"],
        principalDf["PC2"],
        c=colors,
        cmap='gist_rainbow',
        s=15)

    handles, labels= graph.legend_elements(prop="colors", num=len(targets))
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f' UMAP projection of Compound Clustering with {moa_subset}', fontsize=24)
    plt.legend(handles, targets, prop={'size': 10})
    run["images/PCA"].upload(pca_plot)
    
    print("Starting UMAP")
    umap_n_components = 20
    pca = PCA(n_components= umap_n_components)
    principalComponents = pca.fit_transform(df_singleMoA_train_features)
    principalDf = pd.DataFrame(data = principalComponents)
    umap_neighbors = 15
    umap_min_dist = 0.1
    reducer = umap.UMAP(n_neighbors= umap_neighbors, min_dist = umap_min_dist)

    embedding = reducer.fit_transform(principalDf)

    targets =  [i for i in df_singleMoA_labels["Compound ID"].unique()]
    labels_to_targets = {label: i for i, label in enumerate(targets)}
    colors = [labels_to_targets[label] for label in df_singleMoA_labels["Compound ID"]]
    umap_plot = plt.figure(figsize=(15,10))
    graph = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap='gist_rainbow',
        s=15)
    handles, labels= graph.legend_elements(prop="colors", num=len(targets))
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'UMAP projection of Compound Clustering with {moa_subset}', fontsize=24)
    plt.legend(handles, targets, prop={'size': 10})
    run["images/umap"].upload(umap_plot)
    run["metrics/umap_neighbors"] = umap_neighbors
    run["metrics/umap_min_dist"] = umap_min_dist
    run["metrics/pca_components"] = umap_n_components

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# download csvs with all the data pre split
erik10_file = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
train_filename = 'erik10_clue_train_fold_0.csv'
val_filename = 'erik10_clue_val_fold_0.csv'
test_filename = 'erik10_clue_test_fold_0.csv'

training_set, validation_set, test_set =  load_train_valid_data(erik10_file, train_filename, val_filename, test_filename)



L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set)

variance_thresh = 0
normalize_c = False
file_str = "erik10"
df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = True,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = False,
        data_subset = "erik10")

run = neptune.init_run(project='erik-everett-palm/Tomics-PCA-UMAP', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2N2ZlZjczZi05NmRlLTQ1NjktODM5NS02Y2M4ZTZhYmM2OWQifQ==')
run["parameters/moa_dictionary"] = str(dict_moa)
run["parameters/train_filename"] = file_str
run["parameters/variance_threshold"] = variance_thresh

cmpd_clustering(df_train_features, df_train_labels, "cyclooxygenase inhibitor")