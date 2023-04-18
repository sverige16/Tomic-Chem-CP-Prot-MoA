import torch.nn as nn
import pickle 
import numpy as np


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
from torchsummary import summary
nn._estimator_type = "classifier"
import sys
import optuna
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import (apply_class_weights, accessing_correct_fold_csv_files, create_splits, extract_tprofile,
                                        checking_veracity_of_data, LogScaler, EarlyStopper, val_vs_train_loss,
                                        val_vs_train_accuracy, program_elapsed_time, conf_matrix_and_class_report,
                                        pre_processing, create_terminal_table, upload_to_neptune, dict_splitting_into_tensor,
                                        tprofiles_gc_too_func, set_bool_npy, set_bool_hqdose)

sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/Erik_Models/DWTM')
from DWTM import*
from Data_Processing_Categorical import Datapreprocessing as DataPreprocessingCategorical
from Data_Processing_Numerical import Datapreprocessing as DataPreprocessingNumerical
from Image_Canvas_Creation import ImageDatasetCreation 
from Image_Generate import ImageGenerate

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

  
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

# Prep by adding labels to features with title class
df_train_features["Class"] = df_train_labels
df_val_features["Class"] = df_val_labels
df_test_features["Class"] = df_test_labels

# Concatenating all features and labels
df_total = pd.concat([df_train_features, df_val_features, df_test_features], axis=0)
df_total = df_total[0:10]

df_total.to_csv("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/" + file_name + '_' + 'fold0', index = False)

print(f' The range of the indices for L1000_training is: {df_train_features.index[0]} to {df_train_features.index[-1]}')
df_val_initial_index = df_train_features.index[-1] + 1
df_val_final_index = df_val_initial_index + df_val_features.shape[0] - 1
print(f' The range of the indices for L1000_validation is: {df_val_initial_index} to  {df_val_final_index}')
df_test_initial_index = df_val_final_index + 1
df_test_final_index = df_test_initial_index + df_test_features.shape[0] - 1
print(f' The range of the indices for L1000_test is: {df_test_initial_index} to {df_test_final_index}')
# Save all the labels and moa dictionary in a pickle file and store it with the generated images
index_tracker = {'train': [df_train_features.index[0],df_train_features.index[-1]],
                 'valid': [df_val_initial_index,df_val_final_index],
                 'test': [df_test_initial_index, df_test_final_index]}
labels_moadict = [df_train_labels, df_val_labels, df_test_labels, dict_moa, index_tracker]

with open("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/" + file_name + "labels_moa_dict" +'_' + 'fold0'+ ".pkl", 'wb') as f:
    pickle.dump(labels_moadict, f)

data_path = "/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/" + file_name + '_' + 'fold0'
print("Data Pre-Processing Numerical Data")
PD = DataPreprocessingNumerical(data_path)

r = PD.r_scores()
ImageCanvasCreation = ImageDatasetCreation("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/ProcessedDataset_erik10_hq_8_12.csv") 

feature_no, c = ImageCanvasCreation.pre_preinsertion()

m = 128 #Change according to preference
n= 128  #Change according to preference
#Image size of 128 by 128 is being used

s_list, h_list = ImageCanvasCreation.preinsertion(c,r,m,n,feature_no) 
print("Start Image Generation")
#Path for Processed Dataset
im = ImageGenerate("/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/ProcessedDataset_erik10_hq_8_12.csv")
csv_path = "/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/DWTM/ProcessedDataset_erik10_hq_8_12.csv"
csv_path2 = data_path
df2 = pd.read_csv(csv_path2)
if len(s_list)!=len(h_list):
  h_list.pop()
df = pd.read_csv(csv_path)
df.info()
for i in range(0,df.shape[0]):
  class_name = df.iloc[i]['Class']#.astype(int) #Why type is Int?
  #dmc =df.iloc[i]['Zone Name']
  #da =df.iloc[i]["Date"]
  image_file_name =  str(i)
  #ccc,  data = count_chars_val(d)
  
  data = []
  for i, v in enumerate(df.loc[i].to_list()):
    if i!=2:
      data.append(v)

  img_height = m #Change according to preference, match with m and n of Part 3
  img_width = n #Change according to preference, match with m and n of Part 3
  test_img = im.image_generator(data,class_name=class_name, s_list=s_list, h_list=h_list, img_height=128, img_width=128, image_file_name=image_file_name)
