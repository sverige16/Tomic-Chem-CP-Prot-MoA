## .csv files
all_data.csv: csv file containing information for Tian's 10 moas (compound and paths to the pictures)
fl_data.csv: updated all_data that phil used before sending in for publication. Unclear as to what I should use it for.
dmso_stats: used for normalization of the various plates. Think it is for tian's but unsure
specs935-v1-compounds: compound info for specs 935
specs1k-v2-compounds: compound info for specs 1k-v2
compounds_v1v2.csv: compound info for concatenated specs 935 and specs 1k-v2
cmpds_v1v2_no_enants: compound info for concatenated specs 935 and specs 1k-v2 with enantiomers removed
5_fold_data_sets: directory with data sets that have been split into test and training sets using stratified k-fold

## .pickles
dictionary2.pickle: dictionary for tian's work, in which the key are the compounds for the 10 classes and the value is the smiles strings
## directories
paths_to_channels_creation: Contains the code to create paths_to_channels csvs and the csvs for SPECSv1 and SPECSv2
## ipynb
combine_specs_compounds.ipynb: combines the different compounds from specs-v1 and specs-v2 into one large csv file after adjusting and matching column names.
creates_cmpds_found_in_clue-v1v2: creates csv with compounds found in both the clue and specs-v1 and specs-v2.