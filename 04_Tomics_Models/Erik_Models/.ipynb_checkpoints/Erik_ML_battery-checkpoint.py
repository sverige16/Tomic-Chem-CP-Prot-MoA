#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import glob
import pickle
import random
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime

import torch
import torchvision.models as models
from torchvision import transforms
import pytorch_tabnet
#from pytorch_tabnet.tab_model import TabNetClassifier
#nn._estimator_type = "classifier"

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, log_loss, accuracy_score, f1_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import VotingClassifier

from cmapPy.pandasGEXpress.parse import parse
import cmapPy.pandasGEXpress.subset_gctoo as sg

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import ( accessing_correct_fold_csv_files, create_splits,
                                        checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss,
                                        val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report,
                                        pre_processing, create_terminal_table, upload_to_neptune, dict_splitting_into_tensor,
                                        tprofiles_gc_too_func, set_bool_npy, set_bool_hqdose, accessing_all_folds_csv)

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# -------------------------------------------------------------------------------------------------------------------------


def get_models():
    '''
    Input:
        class weight: including or not including class weight.
    Output:
        A list of tuples, with a str with a descriptor followed by the classifier function)
    '''
    #TNC = TabNetClassifier()
    #TNC._estimator_type = "classifier"
    models = list()
    #models.append(('logreg', LogisticRegression(class_weight = 'balanced', solver= "liblinear", penalty = "l2"))) 
    #models.append(('RFC',RandomForestClassifier())) 
    models.append(('gradboost', GradientBoostingClassifier()))
    #models.append(('Ada', AdaBoostClassifier()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
    models.append(('Bagg',BaggingClassifier()))
    #models.append(('Tab', TNC))
    return models


   
file_name = "erik10_hq_8_12"
for fold_int in range(0,5):
    print(fold_int)
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int)
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

    # Converting labels to numerical values
    extract_index = lambda x: pd.Series(dict_moa[x]).idxmax()
    df_train_labels = df_train_labels["moa"].apply(extract_index)
    df_val_labels = df_val_labels["moa"].apply(extract_index)
    df_test_labels = df_test_labels["moa"].apply(extract_index)


    ensemble = False
    models = get_models()
    scores = list()
    yn_class_weights = 'False'
        # battery of classifiers
    for class_alg in models:
        start = time.time()
        model_name = class_alg[0]
        print(f'Running {model_name} model. ')
    
        classifier = class_alg[1]
        classifier.fit(df_train_features.values, df_train_labels)
        all_predictions = classifier.predict(df_test_features.values)
        f1_score_from_model = f1_score(df_test_labels, all_predictions, average= "macro") 
        scores.append(f1_score_from_model)
        end = time.time()
        
        elapsed_time = program_elapsed_time(start, end)
        create_terminal_table(elapsed_time, df_test_labels, all_predictions)
        upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = normalize_c,
                        yn_class_weights = yn_class_weights, 
                        elapsed_time = elapsed_time, 
                        all_labels = df_test_labels.values,
                        all_predictions = all_predictions,
                        dict_moa = dict_moa)
        
    if ensemble:
            # 'soft':  predict the class labels based on the predicted probabilities p for classifier 
        start = time.time()
        ensemble = VotingClassifier(estimators = models, voting = 'soft', weights = scores)
        ensemble.fit(df_train_features.values, df_train_labels)
        all_predictions = ensemble.predict(df_test_features.values)
        end = time.time()
        
        elapsed_time = program_elapsed_time(start, end)
        create_terminal_table(elapsed_time, df_test_labels, all_predictions)
        upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = str(models),
                        normalize = normalize_c,
                        yn_class_weights = yn_class_weights, 
                        elapsed_time = elapsed_time, 
                        all_labels = df_test_labels.values,
                        all_predictions = all_predictions,
                        dict_moa = dict_moa)
    
    '''
    file_name = "erik10_hq_8_12"
    #file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
    training_set, validation_set, test_set =  accessing_correct_fold_csv_files(file_name)
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    variance_thresh = 0.8
    normalize_c = 'True'
    npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c)
    df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
            clue_gene, 
            npy_exists = npy_exists,
            use_variance_threshold = variance_thresh, 
            normalize = normalize_c, 
            save_npy = save_npy,
            data_subset = file_name)
    checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)

    # Converting labels to numerical values
    extract_index = lambda x: pd.Series(dict_moa[x]).idxmax()
    df_train_labels = df_train_labels["moa"].apply(extract_index)
    df_val_labels = df_val_labels["moa"].apply(extract_index)
    df_test_labels = df_test_labels["moa"].apply(extract_index)


    ensemble = False
    models = get_models()
    scores = list()
    yn_class_weights = 'False'
        # battery of classifiers
    for class_alg in models:
        start = time.time()
        model_name = class_alg[0]
        print(f'Running {model_name} model. ')
    
        classifier = class_alg[1]
        classifier.fit(df_train_features.values, df_train_labels)
        all_predictions = classifier.predict(df_test_features.values)
        f1_score_from_model = f1_score(df_test_labels, all_predictions, average= "macro") 
        scores.append(f1_score_from_model)
        end = time.time()
        
        elapsed_time = program_elapsed_time(start, end)
        create_terminal_table(elapsed_time, df_test_labels, all_predictions)
        upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = model_name,
                        normalize = normalize_c,
                        yn_class_weights = yn_class_weights, 
                        elapsed_time = elapsed_time, 
                        all_labels = df_test_labels.values,
                        all_predictions = all_predictions,
                        dict_moa = dict_moa)
        
    if ensemble:
            # 'soft':  predict the class labels based on the predicted probabilities p for classifier 
        start = time.time()
        ensemble = VotingClassifier(estimators = models, voting = 'soft', weights = scores)
        ensemble.fit(df_train_features.values, df_train_labels)
        all_predictions = ensemble.predict(df_test_features.values)
        end = time.time()
        
        elapsed_time = program_elapsed_time(start, end)
        create_terminal_table(elapsed_time, df_test_labels, all_predictions)
        upload_to_neptune('erik-everett-palm/Tomics-Models',
                        file_name = file_name,
                        model_name = str(models),
                        normalize = normalize_c,
                        yn_class_weights = yn_class_weights, 
                        elapsed_time = elapsed_time, 
                        all_labels = df_test_labels.values,
                        all_predictions = all_predictions,
                        dict_moa = dict_moa)
    '''