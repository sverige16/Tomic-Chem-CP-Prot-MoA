
#!/usr/bin/env python
# coding: utf-8

# Import Statements
import pandas as pd
import os 
from IGTD_functions import min_max_transform, table_to_image
  # The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.
import datetime
import time
import neptune.new as neptune
import os
import time
from datetime import datetime
import pandas as pd

import sys
sys.path.append('/home/jovyan/Tomics-CP-Chem-MoA/05_Global_Tomics_CP_CStructure/')
from Erik_alll_helper_functions import ( accessing_all_folds_csv,
                                        create_splits, 
                                        checking_veracity_of_data, 
                                        pre_processing, 
                                        set_bool_npy, 
                                        set_bool_hqdose, 
                                        getting_correct_image_indices)


start = time.time()
now = datetime.now()
now = now.strftime("%d_%m_%Y-%H:%M:%S")
print("Begin Training")
model_name = "IGTD"

# Downloading all relevant data frames and csv files ----------------------------------------------------------

# clue column metadata with columns representing compounds in common with SPECs 1 & 2
clue_sig_in_SPECS = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_sig_in_SPECS1&2.csv', delimiter = ",")

# clue row metadata with rows representing transcription levels of specific genes
clue_gene = pd.read_csv('/home/jovyan/Tomics-CP-Chem-MoA/04_Tomics_Models/init_data_expl/clue_geneinfo_beta.txt', delimiter = "\t")

file_name = "erik10_hq_8_12"
#file_name = input("Enter file name to investigate: (Options: tian10, erik10, erik10_hq, erik10_8_12, erik10_hq_8_12, cyc_adr, cyc_dop): ")
training_set, validation_set, test_set =  accessing_all_folds_csv(file_name, 0)
hq, dose = set_bool_hqdose(file_name)
L1000_training, L1000_validation, L1000_test = create_splits(training_set, validation_set, test_set, hq = hq, dose = dose)
checking_veracity_of_data(file_name, L1000_training, L1000_validation, L1000_test)
variance_thresh = 0
normalize_c = False
npy_exists, save_npy = set_bool_npy(variance_thresh, normalize_c, five_fold = 'True')
df_train_features, df_val_features, df_train_labels, df_val_labels, df_test_features, df_test_labels, dict_moa = pre_processing(L1000_training, L1000_validation, L1000_test, 
        clue_gene, 
        npy_exists = npy_exists,
        use_variance_threshold = variance_thresh, 
        normalize = normalize_c, 
        save_npy = save_npy,
        data_subset = file_name)
checking_veracity_of_data(file_name, df_train_labels, df_val_labels, df_test_labels)


num_row = 6    # Number of pixel rows in image representation
num_col = 163    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.


'''
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
'''

# entire data set
df_total = pd.concat([df_train_features, df_val_features, df_test_features], axis=0).reset_index(drop=True)
getting_correct_image_indices(model_name, file_name, df_total)
df_total.to_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/IGTD_erik10_hq_8_12_fold0.pkl')
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
result_dir = '/scratch2-shared/erikep/Results/'  + file_name + '_' + fea_dist_method
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
result_dir = '/scratch2-shared/erikep/Results/' + file_name + '_' + fea_dist_method 
os.makedirs(name=result_dir, exist_ok=True)

print("running table to image")
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
