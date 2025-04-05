import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from dtuimldmtools import rlr_validate
from sklearn import model_selection
from load_data import X_normalized, y2
from dtuimldmtools import draw_neural_net, train_neural_net
from ann_validate import *
from tqdm import tqdm  # For progress bars

N, M = X_normalized.shape
K = 5
CV = model_selection.KFold(K, shuffle=True)

Error_test_ANN = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_test_baseline = np.empty((K, 1))
opt_lamdas = np.empty([K,1])
opt_hs = np.empty([K,1])

lambdas = np.logspace(-2, 7, 100)
hs = np.arange(1,6,1)
w_rlr = np.empty((M+1, K))

k = 0
for train_index, test_index in tqdm(CV.split(X_normalized, y2),total = K, desc="Outer CV folds"):
    X_train = X_normalized[train_index]
    y_train = y2[train_index]
    X_test = X_normalized[test_index]
    y_test = y2[test_index]
    internal_cross_validation = 5
    X_train_rlr = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test_rlr = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
    (opt_val_err_rlr,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train_rlr, y_train, lambdas, internal_cross_validation)

    (opt_val_err_ann,
        opt_h,
        test_err_vs_h) = ann_validate(X_train, y_train, hs, internal_cross_validation)

    Xty = X_train_rlr.T @ y_train
    XtX = X_train_rlr.T @ X_train_rlr

    # Compute mean squared error without using the input data at all
    Error_test_baseline[k] = (
        np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
    )

    # Estimating weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M+1)
    lambdaI[0, 0] = 0
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()

    Error_test_rlr[k] = (
        np.square(y_test - X_test_rlr @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
    )
    opt_lamdas[k] = opt_lambda

    X_train_ann = torch.Tensor(X_train)
    y_train_ann = torch.Tensor(y_train)
    X_test_ann = torch.Tensor(X_test)
    y_test_ann = torch.Tensor(y_test)
    y_train_ann = y_train_ann.view(-1, 1)
    y_test_ann  = y_test_ann.view(-1, 1)

    model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, opt_h),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(opt_h, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output")
            )
    loss_fn = torch.nn.MSELoss()

    net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train_ann,
        y=y_train_ann,
        n_replicates=1,
        max_iter=10000,
    )
    
    y_test_est = net(X_test_ann)

    se = (y_test_est.float() - y_test_ann.float()) ** 2  # squared error
    mse = torch.mean(se).item()  # mean
    Error_test_ANN[k] = mse
    opt_hs[k] = opt_h

    k += 1

df_results = pd.DataFrame({
    'fold': range(1, K+1),
    'opt_h': opt_hs.ravel(),
    'ANN_test_error': Error_test_ANN.ravel(),
    'opt_lambda': opt_lamdas.ravel(),
    'RLR_test_error': Error_test_rlr.ravel(),
    'baseline_test_error': Error_test_baseline.ravel()
})
print(df_results)
