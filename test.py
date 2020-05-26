import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import vocab
import utils
import re
from nltk.stem.api import StemmerI
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

"""
with open('config.yml', 'r') as f:
    doc = yaml.safe_load(f)
    print(doc)





def set_epoch(epoch):
    file_name = './config.yml'
    with open(file_name, 'r') as f:
        doc = yaml.safe_load(f)
    doc['train_model']['start_epoch'] = epoch
    with open(file_name, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)


set_epoch(31)
"""

s = str(-123)
l = [ch for ch in s]
print(l)
