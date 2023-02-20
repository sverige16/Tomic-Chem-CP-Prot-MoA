# E_training_test_split
E_training_test_split.ipynb
E_training_test_split.py: Splits data set into csvs. Can split into train and test, or train, test and validate. Also can determine which cell lines/thesholds on cell lines to use.
# ML_battery_L1000
ML_battery_L1000_dim_reduc.py: Finds the scores based on input for pre-processing.
ML_battery_L1000_optuna.py : Battery of test where neptune.ai functions as well optuna. Used to find the optimal hyperparameters given pre-processing.
ML_battery_L1000.py : Deprecated code. Shuffling occurs incorrectly, resulting in no learning going.
# feature_selection_dim_reduc
feature_selection_dim_reduc.py : Produces PCA and UMAP representations when you feed in a CSV with data. Is saved to a project on Neptune.ai
feature_selection_dim_reduc.ipynb: Notebook version
feat_select.ipynb: Developing OneHotEncoding function for 1D_CNN and Simple NN. Also figuring out how to extract the most important features from the linear algorithms to see if that helps in dimensionality reduction.
# 1D_CNN
Erik_1D_CNN_L1000.ipynb : jupyter notebook version
Erik_1D_CNN_L1000.py : 1D_CNN copied from Kaggle competition. Implemented which can now upload results to neptune.ai
# SimpleNN
Erik_SimpleNN_L1000.ipynb: jupyter notebook version
Erik_SimpleNN_L1000.py: Simple NN copied from Kaggle competition. Implemented which can now upload results to neptune.ai.

# Combining_1DCNN_SimpleNN_TabNet
Combining_1DCNN_SimpleNN_TabNet.ipynb: Downloads the TabNet, 1D_CNN and SimpleNN class probabilities and then using those probabilities makes an ensemble prediction.
# KNN
### both used to build up functional transcriptomics generators
Actual_KNN_1000.ipynb: Deprecated
KNN_1000.ipynb: Deprecated

# umap_featreduc_pipeline_optimiz
umap_featreduc_pipeline_optimiz.py: Test implementing optuna, using umap as feature selector and SVD as predictor.