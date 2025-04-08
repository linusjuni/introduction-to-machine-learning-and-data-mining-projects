import numpy as np
from sklearn import model_selection
from tqdm import tqdm
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score

def rlogr_validate(X, y, lambdas, cvf):
    CV = model_selection.KFold(cvf, shuffle=True)
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()

    for train_index, test_index in tqdm(CV.split(X, y),total = cvf,desc="RLOGR optimization"):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
        for l in range(0, len(lambdas)):
            C_value = 1.0 / lambdas[l]

            model = lm.LogisticRegression(penalty='l2', C=C_value, solver='lbfgs', max_iter=1000,fit_intercept=False)
            model.fit(X_train, y_train)
            
            y_est = model.predict(X_test)

            error = 1 - accuracy_score(y_test, y_est)
            test_error[f,l] = error


        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    test_err_vs_lambda = np.mean(test_error, axis=0)

    return (
        opt_val_err,
        opt_lambda,
        test_err_vs_lambda
    )