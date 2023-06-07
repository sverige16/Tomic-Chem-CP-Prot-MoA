#!/usr/bin/env python
# coding: utf-8

# Import Statements
import optuna
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
import pandas as pd
from sklearn.metrics import  f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (create_splits,
                                        pre_processing,
                                         set_bool_npy,
                                        set_bool_hqdose, 
                                        accessing_all_folds_csv)


'''
# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")
'''
# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")
   
def ML_optuna_trials(trial, model_name, df_train_features, df_train_labels, df_test_features, df_test_labels):

    print(f'Running {model_name} model. ')
    if model_name == 'KNN':
        n_neighbors = trial.suggest_int("n_neighbors", 1, 100)
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        leaf_size = trial.suggest_int("leaf_size", 1, 100)
        classifier = KNeighborsClassifier(n_neighbors = n_neighbors, 
                                            algorithm = algorithm, 
                                            leaf_size = leaf_size,
                                            n_jobs=-1)
    elif model_name == 'gradboost':
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        classifier = GradientBoostingClassifier(
                                                learning_rate = learning_rate, 
                                                n_estimators = n_estimators, 
                                                max_depth = max_depth, 
                                               )
    elif model_name == 'Bagg':
        # for some reason, optuna doesn't like the BaggingClassifier() function
        # even when using the same parameters as default, the score is much lower
        # so I'm using the parameters from the default function
        max_samples = trial.suggest_int("max_samples", 1, 10)
        max_features = trial.suggest_int("max_features", 1, 10)
        classifier = BaggingClassifier(max_samples = max_samples,
                                        max_features = max_features,
                                        n_jobs=-1)
        classifier = BaggingClassifier()
    else:
        ValueError('Model not found')
    classifier.fit(df_train_features.values, df_train_labels)
    all_predictions = classifier.predict(df_test_features.values)
    f1_score_from_model = f1_score(df_test_labels, all_predictions, average= "macro")

    return f1_score_from_model

model_name = 'gradboost'
file_name = "erik10_hq_8_12"
for fold_int in range(0,1):
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

storage = 'sqlite:///' + model_name + '.db'
study = optuna.create_study(direction='maximize',
                            storage = storage)
study.optimize(lambda trial: ML_optuna_trials(trial, model_name=model_name, 
                                      df_train_features=df_train_features, 
                                      df_train_labels=df_train_labels, 
                                      df_test_features=df_val_features,
                                      df_test_labels=df_val_labels.values
                                      ), 
                                      n_trials = 25)

#-------------------------------- Writing interesting info into terminal ------------------------# 

print("Number of finished trials: {}".format(len(study.trials)))
print(study.best_params)
print(study.best_value)

f = open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/random/" + model_name + '_' +'_best_params.txt',"w")
# write file
f.write(model_name)
f.write("Best Parameters: " + str(study.best_params))
f.write("Best Value: " + str(study.best_value))
# close file
f.close()