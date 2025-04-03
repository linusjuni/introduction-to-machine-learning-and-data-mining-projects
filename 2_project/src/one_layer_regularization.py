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
lambdas = np.logspace(-2, 7, 50)

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
print(test_err_vs_lambda)