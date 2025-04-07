import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def knearest_validate(X, y, nbrs, cvf):
    CV = model_selection.KFold(cvf, shuffle=True)
    test_error = np.empty((cvf, nbrs))
    y = y.squeeze()

    f = 0
    for train_index, test_index in CV.split(X, y):
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0)
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        for l in range(1, nbrs + 1):
            knclassifier = KNeighborsClassifier(n_neighbors=l)
            knclassifier.fit(X_train, y_train)
            y_est = knclassifier.predict(X_test)

            error = 1 - accuracy_score(y_test, y_est)
            test_error[f,l-1] = error

        f += 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_nbrs = np.argmin(np.mean(test_error, axis=0)) + 1
    test_err_vs_nbrs = np.mean(test_error, axis=0)

    return (
        opt_val_err,
        opt_nbrs,
        test_err_vs_nbrs
    )