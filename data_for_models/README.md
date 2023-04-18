## .csv files
Phil_all_data.csv: csv file containing information for Tian's 10 moas (compound and paths to the pictures)
Phil_fl_data.csv: updated all_data that phil used before sending in for publication. Unclear as to what I should use it for.
Phil_dmso_stats: used for normalization of the various plates. Think it is for tian's but unsure
specs935-v1-compounds: compound info for specs 935
specs1k-v2-compounds: compound info for specs 1k-v2
compounds_v1v2.csv: compound info for concatenated specs 935 and specs 1k-v2
cmpds_v1v2_no_enants: compound info for concatenated specs935 and specs 1k-v2 with enantiomers removed

## .pickles
Phil_dictionary2.pickle: dictionary for tian's work, in which the key are the compounds for the 10 classes and the value is the smiles strings
## directories
paths_to_channels_creation: Contains the code to create paths_to_channels csvs and the csvs for SPECSv1 and SPECSv2
random: place where I place the confusion matrices and classification file before sending it up to neptune.ai
5_fold_dataset: directory with subdirectories that hold all of the 5 fold splits for each of the datasets that I am working with.

## ipynb
combine_specs_compounds.ipynb: combines the different compounds from specs-v1 and specs-v2 into one large csv file after adjusting and matching column names.
creates_cmpds_found_in_clue-v1v2: creates csv with compounds found in both the clue and specs-v1 and specs-v2.
five_way_split.ipynb: the jupyter notebook that was used to build five_way_split.py.
## .py
five_way_split.py: the script that used to divide the various data sets into five splits using sklearn stratified k fold.
bid_five_way_split: include batch ID in the five splits in order to take the enantiomers into account.
