



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
from rdkit import Chem
from rdkit.Chem import DataStructs, AllChem
import torchvision.transforms.functional as TF
from efficientnet_pytorch import EfficientNet 

# A function changing SMILES to Morgan fingerprints 
def smiles_to_array(smiles):
    molecules = Chem.MolFromSmiles(smiles) 
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(molecules, 2)
    x_array = []
    arrays = np.zeros(0,)
    DataStructs.ConvertToNumpyArray(fingerprints, arrays)
    x_array.append(arrays)
    x_array = np.asarray(x_array)
    x_array = ((np.squeeze(x_array)).astype(int)) 
    x_array = torch.from_numpy(x_array)
    return x_array                  

# not deep   50x50
class DeepInsight_Model(nn.Module):
    def __init__(self):
        super(DeepInsight_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=64*25*25, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class Chem_Dataset(torch.utils.data.Dataset):
    def __init__(self, compound_df, labels_df, dict_moa, transform=None):
        self.compound_labels = labels_df    # the entire length of the correct classes that we are trying to predict
        # print(self.img_labels)
        self.compound_df = compound_df        # list of indexes that are a part of training, validation, tes sets
        self.transform = transform       # any transformations done
        self.dict_moa = dict_moa

    def __len__(self):
        ''' The number of data points'''
        return len(self.compound_labels)      

    def __getitem__(self, idx):
        '''Retrieving the compound '''
        #print(idx)
        smile_string = self.compound_df["SMILES"][idx]      # returns smiles by using compound as keys
        #print(smile_string)
        compound_array = smiles_to_array(smile_string)
        #print(f' return from function: {compound}')
        #print(f' matrix: {compound_array}')
        label = self.compound_labels.iloc[idx]             # extract classification using index
        #print(f' label: {label}')
        #label = torch.tensor(label, dtype=torch.float)
        label_tensor = torch.from_numpy(self.dict_moa[label])                  # convert label to number
        if self.transform:                         # uses Albumentations image pipeline to return an augmented image
            compound = self.transform(compound)
        return compound_array.float(), label_tensor.long() # returns the image and the correct label

class Reducer_profiles(torch.utils.data.Dataset):
    def __init__(self, X, y, dict_moa):
        self.X = X
        self.y = y
        self.dict_moa = dict_moa

    def __getitem__(self, index):
        label = self.dict_moa[self.y[index]]
        return self.X[index], torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return len(self.X)


'''
class Chem_Model(nn.Module):
    def __init__(self):
        super(Chem_Model, self).__init__()
        self.Linear1 = nn.Linear(2048, 128)
        self.Linear2 = nn.Linear(128, 64)
        self.Linear3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.Linear3(x)
        return x
'''


class image_network(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.base_model = EfficientNet.from_name('efficientnet-b1', include_top=False, in_channels = 5)
        self.dropout_1 = nn.Dropout(p = 0.3)
        self.Linear_last = nn.Linear(1280, num_classes)
        # self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.dropout_1(self.base_model(x))
        out = out.view(-1, 1280)
        out = self.Linear_last(out)
        # out = self.softmax(out) # don't need softmax when using CrossEntropyLoss
        return out
 
class MyRotationTransform:
    " Rotate by one of the given angles"
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image):
        angle = random.choice(self.angle)
        return TF.rotate(image, angle)
