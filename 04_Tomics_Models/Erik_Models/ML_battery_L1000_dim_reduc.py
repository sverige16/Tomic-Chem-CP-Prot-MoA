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

# Torch
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
# Neptune
import neptune.new as neptune


from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve,log_loss, accuracy_score, f1_score, classification_report
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

def variance_threshold(x_train, x_val, var_threshold ):
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
    x_train = x_train.loc[:,var_thresh.variances_ > var_threshold] # locate all variance thresholds above 0.8, keep those columns
    x_val = x_val.loc[:,var_thresh.variances_ > var_threshold]
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
    #norm_model = StandardScaler()
    trn_norm = pd.DataFrame(norm_model.fit_transform(trn),index = trn.index,columns = trn.columns)
    tst_norm = pd.DataFrame(norm_model.transform(test),index = test.index,columns = test.columns)
    return trn_norm, tst_norm, str(norm_model)


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
        #npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_my10_train.npy', allow_pickle=True)
        npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_cyc_adrtrain.npy', allow_pickle=True)
    elif dataset == 'val':
        #filename = input('Give name of npy file (str): ')
        #npy_set = np.load(path + filename)
        #npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_my10_val.npy', allow_pickle=True)
        npy_set = np.load('/scratch2-shared/erikep/data_splits_npy/nv_cyc_adrval.npy', allow_pickle=True)
    else:
        filename =  input('Give name of npy file (str): ')
        npy_set = np.load(path + filename)
    df = pd.DataFrame(npy_set)
    return df.set_axis([*df.columns[:-1], 'sig_id'], axis=1, inplace=False)

def save_npy(dataset, split_type):
    '''Save the numpy array of the selected transcriptomics profiles
    Input:
        dataset: the numpy array to be saved
    '''
    path = '/scratch2-shared/erikep/data_splits_npy/'
    file_name = input("Give filename for numpy array: ")
    np.save(path + file_name + '_' + split_type, dataset)

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
    models.append(("LDAC",  LinearDiscriminantAnalysis()))
    #models.append(('QDA', QuadraticDiscriminantAnalysis()))
    models.append(('Ridge', RidgeClassifierCV(class_weight = class_weight)))
    #models.append(('RFC',RandomForestClassifier(class_weight= class_weight))) 
    models.append(('gradboost', GradientBoostingClassifier()))
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
    labels_unique = np.unique(labels_val)
    anders = f1_score(labels_val, predictions, labels = labels_unique.tolist(), average = 'macro')
    accuracy = accuracy_score(labels_val, predictions)
    print(f' Accuracy score: {accuracy}')
    print(f' F1 Score: {     anders   }')
    cf_matrix = confusion_matrix(labels_val, predictions)
    print(f' Confusion Matrix: {cf_matrix}')
    plt.figure()
    sns.heatmap(cf_matrix, annot = True, fmt='d').set(title = class_alg[0] + ' Confusion Matrix')
    plt.savefig("Conf_matrix.png")
   
    class_report = classification_report(labels_val, predictions)
    print(class_report)
    f = open("class_info.txt","w")
    # write file
    f.write( str(class_report) )
    # close file
    f.close()
    print('----------------------------------------------------------------------')
    return anders, accuracy
def write_list(a_list, file_type):
    with open('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/pickles/tabnet_pickles/' + file_type + '.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
    print('Done writing binary file')

def save_val(a_list, file_type):
    with open('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/pickles/val_order_pickles/' + file_type + '.pickle', 'wb') as fp:
        pickle.dump(a_list, fp)
    print('Done writing binary file')

# ------------------------------------------------------------------------------------------------------------------------------
def main(train_filename, L1000_training, L1000_validation, 
         clue_gene, npy_exists, apply_class_weight = False, 
         use_variance_threshold = 0, normalize = False, 
         ensemble = False,
         feat_sel = 0):
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
    
    print("extracting training transcriptomes")
    profiles_gc_too_train = tprofiles_gc_too_func(L1000_training, clue_gene)
    if npy_exists:
        df_train = acquire_npy('train')
    else:    
        df_train = np_array_transform(profiles_gc_too_train)
        #save_npy(df_train, "train")
    
    #
    print("extracting validation transcriptomes") 
    profiles_gc_too_valid = tprofiles_gc_too_func(L1000_validation, clue_gene)
    if npy_exists:
        df_val = acquire_npy('val')
    else:    
        df_val = np_array_transform(profiles_gc_too_valid)
        #save_npy(df_val, "val")
    

   
    # merging the transcriptomic profiles with the corresponding MoA class using the sig_id as a key
    df_train = pd.merge(df_train, L1000_training[["sig_id", "moa"]], how = "outer", on ="sig_id")
    df_val = pd.merge(df_val, L1000_validation[["sig_id", "moa"]], how = "outer", on ="sig_id")
    # dropping the sig_id column
    df_train.drop(columns = ["sig_id"], inplace = True)
    df_val.drop(columns = ["sig_id"], inplace = True)
     # separating the features from the labels
    df_train_features = df_train[df_train.columns[: -1]]
    df_val_features = df_val[df_val.columns[: -1]]
    df_train_labels = df_train["moa"]
    df_val_labels = df_val["moa"]
    
  
    df_train_features, df_val_features = feature_selection(df_train_features, df_val_features, feat_sel)

     # to normalize
    if normalize:
        df_train_features, df_val_features, norm_type = normalize_func(df_train_features, df_val_features)
    else:
        norm_type = None
    # applying class weights
    if apply_class_weight:
        class_weight = "balanced"
    else:
        class_weight = None
    
    # -----------------------------------------------------------------------------------------------------------------------------#
    print("starting modelling")
    models = get_models(class_weight)
    scores = list()
    # battery of classifiers
    for class_alg in models:
        classifier = class_alg[1]
        # use variance threshold
        if use_variance_threshold > 0:
            df_train_features_vs, df_val_features_vs = variance_threshold(df_train_features, df_val_features, use_variance_threshold)
            classifier.fit(df_train_features_vs.values.astype("float64"), df_train_labels.values)
            predictions = classifier.predict(df_val_features_vs.values.astype("float64"))
            input_shape = df_train_features_vs.shape[1]
            '''
            if class_alg[0] == 'Tab':
                save_val(df_val[df_val.columns[-1]].values, 'tab_val')
                class_probs= classifier.predict_proba(df_val_vs.values)
                write_list(predictions, 'predictions')
                write_list(class_probs, 'class_probs')'''
        else:
            classifier.fit(df_train_features.values, df_train_labels.values)
            input_shape = df_train_features.shape[1]
            '''
            if class_alg[0] == 'Tab':
                save_val( df_val[df_val.columns[-1]].values, 'tab_val')
                class_probs = classifier.predict_proba(df_val[df_val.columns[: -1]].values)
                write_list(predictions, 'predictions')
                write_list(class_probs, 'class_probs')
                '''
            predictions = classifier.predict(df_val_features.values)
        f1_score_from_model = f1_score(df_val_labels, predictions, average= "macro") 
        scores.append(f1_score_from_model)
        run = neptune.init_run(project='erik-everett-palm/Tomics-Models')
        run['model'] = class_alg[0]
        run["filename"] = train_filename
        run["feat_selec/feat_sel"] = feat_sel
        run['parameters/normalize'] = norm_type
        run['parameters/class_weight'] = class_weight
        run['parameters/use_variance_threshold'] = use_variance_threshold
        run['feat_selec/input_shape'] = input_shape
        if class_alg[0] == 'Ridge':
            run['parameters/alpha'] = classifier.alpha_
            run['feat_selec/n_features_in'] =  classifier.n_features_in_ 
        f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
        run['metrics/f1_score'] = f1_score_p
        run['metrics/accuracy'] = accuracy_p
        import matplotlib.image as mpimg
        conf_img = mpimg.imread('Conf_matrix.png')
        run["files/classification_info"].upload("class_info.txt")
        run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)
        run.stop()
    if ensemble:
        # 'soft':  predict the class labels based on the predicted probabilities p for classifier 
        ensemble = VotingClassifier(estimators = models, voting = 'soft', weights = scores)
        
        if use_variance_threshold:
                ensemble.fit(df_train_features_vs.values, df_train_labels.values)
                predictions = ensemble.predict(df_val_features_vs.values)

        else:
            ensemble.fit(df_train_features.values, df_train_labels.values)
            predictions = ensemble.predict(df_val.values)
        
        run = neptune.init_run(project='erik-everett-palm/Tomics-Models')
        run['model'] = str(models)
        run["feat_selec/feat_sel"] = feat_sel
        run["filename"] = train_filename
        run['parameters/normalize'] = norm_type
        run['parameters/class_weight'] = class_weight
        run['parameters/use_variance_threshold'] = use_variance_threshold
        f1_score_p, accuracy_p = printing_results(class_alg, df_val[df_val.columns[-1]].values, predictions)
        run['metrics/f1_score'] = f1_score_p
        run['metrics/accuracy'] = accuracy_p
        import matplotlib.image as mpimg
        conf_img = mpimg.imread('Conf_matrix.png')

        run["files/class_info"].upload("class_info.txt")
        run["images/Conf_matrix.png"] =  neptune.types.File.as_image(conf_img)

if __name__ == "__main__": 
    for i in np.arange(0.5, 1.2, 0.1 ):
        # train_filename = input('Training Data Set Filename: ')
        # valid_filename = input('Validation Data Set Filename: ')
        train_filename = 'L1000_training_set_nv_cyc_adr.csv' # 2
        valid_filename = 'L1000_test_set_nv_cyc_adr.csv'     # 2
        #train_filename = 'L1000_training_set_nv_my10.csv' #10
        #valid_filename = 'L1000_test_set_nv_my10.csv'  #10
        
        L1000_training, L1000_validation =  load_train_valid_data(train_filename, valid_filename)
        main(train_filename,
            L1000_training = L1000_training, 
            L1000_validation = L1000_validation, 
            clue_gene= clue_gene, 
            npy_exists = True,
            apply_class_weight= True,
            use_variance_threshold = i, 
            normalize= False,
            ensemble = False,
            feat_sel= 0)
