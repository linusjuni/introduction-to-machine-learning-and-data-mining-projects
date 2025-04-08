import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns
from dtuimldmtools import rlr_validate
from load_data import X_normalized, y
from rlogr_validate import *

X = np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), 1)
N, M = X.shape

lambdas = np.logspace(-5, 7, 100)

(opt_val_err,
opt_lambda,
test_err_vs_lambda) = rlogr_validate(X, y, lambdas, 10)

C_value = 1.0 / opt_lambda

model = lm.LogisticRegression(penalty='l2', C=C_value, solver='lbfgs', max_iter=1000,fit_intercept=False)
model.fit(X, y)
weights = model.coef_.flatten()

plt.figure(figsize=(12, 6))
feature_names = ['Offset', 'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
plt.barh(range(len(weights)), weights)
plt.yticks(range(len(weights)), feature_names)
plt.xlabel('Coefficient Value')
plt.title('Average Feature Weights in Ridge Regression')
plt.grid(axis='x')
plt.tight_layout()
plt.show()