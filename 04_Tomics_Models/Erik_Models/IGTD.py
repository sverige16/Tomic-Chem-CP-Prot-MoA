import pandas as pd
import os 
from IGTD_functions import min_max_transform, table_to_image

#!/usr/bin/env python
# coding: utf-8

# In[16]:


from pyDeepInsight import ImageTransformer, Norm2Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


# In[42]:


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
import pickle

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

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import *


start = time.time()
now = datetime.datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

'''
 # download csvs with all the data pre split
#cyc_adr_file = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/cyc_adr/'
#train_filename = 'cyc_adr_clue_train_fold_0.csv'
#val_filename = 'cyc_adr_clue_val_fold_0.csv'
#test_filename = 'cyc_adr_clue_test_fold_0.csv'
#training_set, validation_set, test_set =  load_train_valid_data(cyc_adr_file, train_filename, val_filename, test_filename)
'''
    
erik10_file = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/5_fold_data_sets/erik10/'
train_filename = 'erik10_clue_hq_train_fold_0.csv'
val_filename = 'erik10_clue_hq_val_fold_0.csv'
test_filename = 'erik10_clue_hq_test_fold_0.csv'
training_set, validation_set, test_set =  load_train_valid_data(erik10_file, train_filename, val_filename, test_filename)

L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = True)

variance_thresh = 0
normalize_c = False
file_str = "erik10_hq"
df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = True,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = False,
        data_subset = file_str)

num_row = 6    # Number of pixel rows in image representation
num_col = 163    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.



# Find indices for generated images
print(df_train_features.shape)
print(df_val_features.shape)
print(df_test_features.shape)

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
with open('/scratch2-shared/erikep/Results/labels_hq_moadict.pkl', 'wb') as f:
    pickle.dump(labels_moadict, f)


# entire data set
df_total = pd.concat([df_train_features, df_val_features, df_test_features], axis=0)
# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
#data = pd.read_csv('../Data/Example_Gene_Expression_Tabular_Data.txt', low_memory=False, sep='\t', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=0)
#data = data.iloc[:, :num]
print("normalizing data")
norm_data = min_max_transform(df_total.values)
norm_data = pd.DataFrame(norm_data, columns = df_total.columns, index = df_total.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.
'''
print("IGTD using Euclidean distance and absolute error")
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = '/scratch2-shared/erikep/Results/Euc_full'
os.makedirs(name=result_dir, exist_ok=True) 

print("running table to image")
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
''' 
# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
print("IGTD using Pearson correlation coefficient and squared error")
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'  
result_dir = '/scratch2-shared/erikep/Results/hq_Pear'
os.makedirs(name=result_dir, exist_ok=True)

print("running table to image")
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
