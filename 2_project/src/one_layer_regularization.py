import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from dtuimldmtools import rlr_validate
from load_data import X, y

X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
N, M = X.shape

lambdas = np.logspace(-2, 7, 100)
w_rlr = np.empty(M)

(opt_val_err,
opt_lambda,
mean_w_vs_lambda,
train_err_vs_lambda,
test_err_vs_lambda,) = rlr_validate(X, y, lambdas, 10)

plt.title("Optimal lambda: 1e{0}".format(np.log10(opt_lambda)))
plt.loglog(
    lambdas, train_err_vs_lambda.T, "b.-", lambdas, test_err_vs_lambda.T, "r.-"
)
plt.xlabel("Regularization factor")
plt.ylabel("Squared error (crossvalidation)")
plt.legend(["Train error", "Validation error"])
plt.grid()
plt.show()

lambdaI = opt_lambda * np.eye(M)
lambdaI[0, 0] = 0 
w_rlr[:] = np.linalg.solve((X.T @ X) + lambdaI, X.T @ y).squeeze()

