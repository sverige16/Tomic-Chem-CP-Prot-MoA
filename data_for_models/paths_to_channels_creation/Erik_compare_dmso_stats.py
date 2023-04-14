#!/usr/bin/env python
# coding: utf-8

# In[1]:

'''
This code dubbelchecks that the two methods (1. using stats.recorder and 2. doing mean and std. in one fell swoop) of calculating the mean and std of the dmso images are the same.
'''

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from functions import *
import tqdm
import stats_recorder as stat_rec

# download CSV path with dmso paths
dmso_csv_path = '/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/paths_channels_dmso_v1v2.csv'
df_dmso = pd.read_csv(dmso_csv_path)

# extract all rows with dmso
df_dmso = df_dmso[df_dmso["neg_control_type"] == '[dmso]']

# extract all unique plates
all_plates = df_dmso.plate.unique()
all_plates = all_plates[0:2]


dmso_stats1 = {}
dmso_stats2 = {}
for plate in tqdm.tqdm(all_plates):
    dmso_stats1[plate] = {}
    dmso_stats2[plate] = {}
    rows = df_dmso[df_dmso.plate == plate]
    #if rows.shape[0] > 300:
        #rows = rows.sample(n= 300, random_state=42)
    for c in tqdm.tqdm(['C1','C2','C3','C4','C5']):
        mystats = stat_rec.StatsRecorder()
        df_split = np.array_split(rows, 3)
        for i in range(0, len(df_split)):
            sub_sample = df_split[i]
            im = []
            for i in tqdm.tqdm(range(len(sub_sample))):
                path = sub_sample[c].iloc[i]
                try:
                    im.append(cv2.imread(path, -1))
                except:
                    print('error with path: ', path)
            print('shape of im: ', np.shape(im))
            mystats.update(im)
        dmso_stats1[plate][c] = {'m': mystats.mean.mean(), 'std': mystats.std.mean()}

        im = []
        for i in tqdm.tqdm(range(len(rows))):
            path = rows[c].iloc[i]
            try:
                im.append(cv2.imread(path, -1))
            except:
                print('error with path: ', path)
        print('shape of im: ', np.shape(im))
        dmso_stats2[plate][c] = {'m': np.mean(im), 'std':np.std(im)}
assert dmso_stats1 == dmso_stats2, 'The two methods of calculating the mean and std of the dmso images are not the same.'
"""
reform = {(outerKey, innerKey): values for outerKey, innerDict in dmso_stats.items() for innerKey, values in innerDict.items()}


df_test = pd.DataFrame(reform)

df_test.to_csv('/home/jovyan/Tomics-CP-Chem-MoA/data_for_models/paths_to_channels_creation/dmso_stats_v1v2.csv')

"""

