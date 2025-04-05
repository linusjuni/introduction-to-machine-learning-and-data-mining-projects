import torch
import numpy as np
from dtuimldmtools import draw_neural_net, train_neural_net
from sklearn import model_selection
from tqdm import tqdm  # For progress bars

def ann_validate(X, y, hs, cvf):
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    test_error = np.empty((cvf, len(hs)))
    f = 0
    y = y.squeeze()

    for train_index, test_index in tqdm(CV.split(X, y),total = cvf,desc="ANN optimization"):
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])
        y_train = y_train.view(-1, 1)
        y_test  = y_test.view(-1, 1)
        
        mu = torch.mean(X_train, dim=0)
        sigma = torch.std(X_train, dim=0)
        X_train = (X_train - mu) / sigma
        X_test  = (X_test  - mu) / sigma

        for l in range(0, len(hs)):
            model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, hs[l]),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(hs[l], 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output")
            )
            loss_fn = torch.nn.MSELoss()
            
            net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=X_train,
            y=y_train,
            n_replicates=1,
            max_iter=10000,)
            
            y_test_est = net(X_test)

            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = torch.mean(se).item()  # mean
            test_error[f,l] = mse

        f = f + 1

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_h = hs[np.argmin(np.mean(test_error, axis=0))]
    test_err_vs_h = np.mean(test_error, axis=0)

    return (
        opt_val_err,
        opt_h,
        test_err_vs_h
    )