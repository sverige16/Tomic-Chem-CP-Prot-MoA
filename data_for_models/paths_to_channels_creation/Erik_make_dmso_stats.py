#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""
Finds the mean and std of the dmso images for each channel and plate and saves it in a dictionary, which is then converted to a csv.
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from functions import *
import tqdm
import stats_recorder as stat_rec

# download CSV path with dmso paths
dmso_csv_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_dmso_v1v2.csv'
df_dmso = pd.read_csv(dmso_csv_path)

# extract all rows with dmso
df_dmso = df_dmso[df_dmso["neg_control_type"] == '[dmso]']

# extract all unique plates
all_plates = df_dmso.plate.unique()

'''
Pseudocode:
For every plate
1. create pandas
2. find all rows with dmso with that plate number
3. if more than 300 rows
    a. Split into 7 chunks
    b. Perform 3.5 by keeping mean/std per chunk and averaging
3.5 else: For for each channel
    a. extract image
    b. get out the mean and std for each of the pixel values.
4. reformat to save in pandas form
5. save as csv
'''
dmso_stats = {}

for plate in tqdm.tqdm(all_plates):
    dmso_stats[plate] = {}
    rows = df_dmso[df_dmso.plate == plate]
    if rows.shape[0] > 300:
            #rows = rows.sample(n= 300, random_state=42)
        for c in tqdm.tqdm(['C1','C2','C3','C4','C5']):
            mystats = stat_rec.StatsRecorder()
            df_split = np.array_split(rows, 7)
            for i in range(0, len(df_split)):
                sub_sample = df_split[i]
                im = []
                for i in tqdm.tqdm(range(len(sub_sample))):
                    path = sub_sample[c].iloc[i]
                    try:
                        im.append(cv2.imread(path, -1))
                    except:
                        print('error with path: ', path)
                print(np.shape(im))
                mystats.update(im)
            dmso_stats[plate][c] = {'m': mystats.mean.mean(), 'std': mystats.std.mean()}
    else:
        for c in tqdm.tqdm(['C1','C2','C3','C4','C5']):
            im = []
            for i in tqdm.tqdm(range(len(rows))):
                path = rows[c].iloc[i]
                try:
                    im.append(cv2.imread(path, -1))
                except:
                    print('error with path: ', path)
            print(np.shape(im))
            dmso_stats[plate][c] = {'m': np.mean(im), 'std':np.std(im)}

reform = {(outerKey, innerKey): values for outerKey, innerDict in dmso_stats.items() for innerKey, values in innerDict.items()}


df_test = pd.DataFrame(reform)

df_test.to_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')


