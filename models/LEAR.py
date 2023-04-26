from models.create_dataset import create_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# load data
data = create_dataset(target_col='DK1_price')

# split into train and test
test_cutoff = '2021-01-01 00:00:00'
train = data[data.index < test_cutoff]
test = data[data.index >= test_cutoff]

# 20 % of train data is validation data
train, val = train_test_split(train, test_size=0.2, shuffle=False)



