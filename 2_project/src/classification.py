import numpy as np
import pandas as pd
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from sklearn import model_selection
from load_data import X_normalized, y
from ann_validate import *
from tqdm import tqdm
from rlogr_validate import * 
K = 5
CV = model_selection.KFold(K, shuffle=True)

Error_test_logr = np.empty((K, 1))
Error_test_baseline = np.empty((K, 1))
Error_test_ct = np.empty((K, 1))
opt_lamdas = np.empty([K,1])

lambdas = np.logspace(-2, 7, 100)

k = 0
for train_index, test_index in tqdm(CV.split(X_normalized, y),total = K, desc="Outer CV folds"):
    X_train = X_normalized[train_index]
    y_train = y[train_index]
    X_test = X_normalized[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    X_train_rlr = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test_rlr = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
    
    (opt_val_err_ann,
        opt_lambda,
        test_err_vs_lambda) = rlogr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    counts = np.bincount(y_train)
    count_0, count_1 = counts[0], counts[1]
    largest_class = 0
    if count_0 >= count_1: 
        counts_test = np.bincount(y_test)
        count_0,count_1 = counts_test[0], counts_test[1]
        Error_test_baseline[k] = (count_1 / (count_0 + count_1))
    else:
        counts_test = np.bincount(y_test)
        count_0,count_1 = counts_test[0], counts_test[1]
        Error_test_baseline[k] = (count_0 / (count_0 + count_1))


    k = k + 1