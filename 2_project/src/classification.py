import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from dtuimldmtools import rlr_validate
from sklearn import model_selection
from load_data import X_normalized, y
from dtuimldmtools import draw_neural_net, train_neural_net
from ann_validate import *
from tqdm import tqdm

K = 5
CV = model_selection.KFold(K, shuffle=True)

Error_test_logr = np.empty((K, 1))
Error_test_baseline = np.empty((K, 1))
Error_test_ct = np.empty((K, 1))
opt_lamdas = np.empty([K,1])

k = 0
for train_index, test_index in tqdm(CV.split(X_normalized, y),total = K, desc="Outer CV folds"):
    
    k = k + 1