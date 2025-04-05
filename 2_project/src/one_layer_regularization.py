import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from dtuimldmtools import rlr_validate
from load_data import X_normalized, y2

X = np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), 1)
N, M = X.shape

lambdas = np.logspace(-2, 7, 100)
w_rlr = np.empty(M)

(opt_val_err,
opt_lambda,
mean_w_vs_lambda,
train_err_vs_lambda,
test_err_vs_lambda,) = rlr_validate(X, y2, lambdas, 10)

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
w_rlr[:] = np.linalg.solve((X.T @ X) + lambdaI, X.T @ y2).squeeze()
print(w_rlr)
plt.figure(figsize=(12, 6))
feature_names = ['Offset', 'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
plt.barh(range(len(w_rlr)), w_rlr)
plt.yticks(range(len(w_rlr)), feature_names)
plt.xlabel('Coefficient Value')
plt.title('Average Feature Weights in Ridge Regression')
plt.grid(axis='x')
plt.tight_layout()
plt.show()