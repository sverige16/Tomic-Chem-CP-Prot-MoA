
import numpy as np
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import torch
from torch.utils.data import DataLoader


# Create dummy data with class imbalance 99 to 1
class_counts = torch.tensor([104, 642, 784])
numDataPoints = class_counts.sum()
data_dim = 5
bs = 170
data = torch.randn(numDataPoints, data_dim)

target = torch.cat((torch.zeros(class_counts[0], dtype=torch.long),
                    torch.ones(class_counts[1], dtype=torch.long),
                    torch.ones(class_counts[2], dtype=torch.long) * 2))

print('target train 0/1/2: {}/{}/{}'.format(
    (target == 0).sum(), (target == 1).sum(), (target == 2).sum()))

# Compute samples weight (each sample should get its own weight)
class_sample_count = torch.tensor(
    [(target == t).sum() for t in torch.unique(target, sorted=True)])
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in target])

# Create sampler, dataset, loader
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_dataset = torch.utils.data.TensorDataset(data, target)
#train_dataset = triaxial_dataset(data, target)
train_loader = DataLoader(
    train_dataset, batch_size=bs, num_workers=0, sampler=sampler)

# Iterate DataLoader and check class balance for each batch
for i, (x, y) in enumerate(train_loader):
    print("batch index {}, 0/1/2: {}/{}/{}".format(
        i, (y == 0).sum(), (y == 1).sum(), (y == 2).sum()))
