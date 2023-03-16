## .ipynb files
Erik_SQL_Paths.ipynb: notebook that shows step for step the building of the function to create the paths csv/ paths_to_channels.py
combining_paths_channels.ipynb: notebook that combines treated and potentially dmso csvs to do analyses on all sites with images
Erik_make_dmso_stats.ipynb: Erik python notebook used to find mean and standard deviation for each plate. 
## .py files
make_dmso_stats.py: Phil's python file to create stats for image normalization.
paths_to_channels.py: the function version of Erik_SQL_paths, taking in a compound csv returning csv with all the 5 channel images. Some problems with column names across specs 1 and 2. Need to adjust.
## CSV Files
specs1k_v2_paths_dmso: has paths to the images for all 5 channels of certain site with only dmso
specs1k_v2_paths_treated: has paths to the images for all 5 channels of certain site perturbed by a small molecule.
specs935-v1_paths: has paths to the images for all 5 channels of certain site perturbed by a small molecule.
specs935-v1_paths_dmso:  has paths to the images for all 5 channels of certain site with only dmso
dmso_stats_v1v2.csv: the mean and standard deviation  for  each plate, produced by Erik_make_dmso_stats.py
paths_channels_dmso_v1v2.csv: has all paths to all of the images for all 5 channels with only dmso.
paths_channels_treated_v1v2.csv: has paths to the images for all 5 channels of all sites treated with images for specs 1 and 2.

