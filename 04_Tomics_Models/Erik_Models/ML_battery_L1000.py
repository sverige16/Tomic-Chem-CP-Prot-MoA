#!/usr/bin/env python
# coding: utf-8
import sys
import time
import pandas as pd
import torchvision.models as models
from sklearn.metrics import  f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (create_splits,
                                        program_elapsed_time,
                                        pre_processing,
                                        create_terminal_table, 
                                        upload_to_neptune, 
                                         set_bool_npy,
                                        set_bool_hqdose, 
                                        accessing_all_folds_csv)

# Downloading all relevant data frames and csv files ----------------------------------------------------------
# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

# -------------------------------------------------------------------------------------------------------------------------

def get_models(model_to_use):
    '''
    Input:
        model_to_use: a list of strings, each string is a model to use
    Output:
        A list of tuples, with a str with a descriptor followed by the classifier function)         '''
    models = list()
    for i in model_to_use:
        if i == 'logreg':
            models.append(('logreg', LogisticRegression())) 
        elif i == 'RFC':
            models.append(('RFC',RandomForestClassifier())) 
        elif i == 'gradboost':
            models.append(('gradboost', GradientBoostingClassifier(learning_rate = 0.10776446183032215,
                                                                    max_depth= 4, 
                                                                    n_estimators = 160)))
        elif i == 'Ada':
            models.append(('Ada', AdaBoostClassifier()))
        elif i == 'KNN':
            models.append(('KNN', KNeighborsClassifier(algorithm = 'auto',
                                                       
                                                    n_neighbors = 1,
                                                    leaf_size = 36,
                                                    n_jobs = -1 )))
        elif i == 'Bagg':
            models.append(('Bagg',BaggingClassifier(n_jobs=-1)))
        elif i == 'Tab':
            TNC = TabNetClassifier()
            TNC._estimator_type = "classifier"
            models.append(('Tab', TNC))
        else:
            ValueError('Model not found')
    return models


file_name = "erik10_hq_8_12"
for fold_int in range(0,5):
    print(f' Fold Iteration: {fold_int}')
    training_set, validation_set, test_set = accessing_all_folds_csv(file_name, fold_int) # access split compounds
    hq, dose = set_bool_hqdose(file_name)
    L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)

    #checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
    variance_thresh = 0
    normalize_c = False
    npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c, five_fold = 'True')
    df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training,
            L1000_validation = L1000_validation,
            L1000_test = L1000_test, 
            clue_gene = clue_gene, 
            npy_exists = npy_exists,
            use_variance_threshold = variance_thresh, 
            normalize = normalize_c, 
            save_npy = save_npy,
            data_subset = file_name)
    #checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)

    # Converting labels to numerical values
    extract_index = lambda x: pd.Series(dict_moa[x]).idxmax()
    df_train_labels = df_train_labels["moa"].apply(extract_index)
    df_val_labels = df_val_labels["moa"].apply(extract_index)
    df_test_labels = df_test_labels["moa"].apply(extract_index)


    ensemble = False
    models = get_models(['gradboost'])
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
    