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
from knearest_validate import * 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true_knn = []
y_pred_knn = []
y_true_logr = []
y_pred_logr = []

K = 10
CV = model_selection.KFold(K, shuffle=True)

Error_test_logr = np.empty((K, 1))
Error_test_baseline = np.empty((K, 1))
Error_test_knn = np.empty((K, 1))

opt_lambdas = np.empty([K,1])
opt_ns = np.empty([K,1])

lambdas = np.logspace(-2, 7, 100)
nbrs_range = 25
k = 0
for train_index, test_index in tqdm(CV.split(X_normalized, y),total = K, desc="Outer CV folds"):
    X_train = X_normalized[train_index]
    y_train = y[train_index]
    X_test = X_normalized[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    X_train_rlogr = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_test_rlogr = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), 1)
    
    #Inner k-fold for finding optimal lambda
    (opt_val_err_rlogr,
        opt_lambda,
        test_err_vs_lambda) = rlogr_validate(X_train_rlogr, y_train, lambdas, internal_cross_validation)
    opt_lambdas[k] = opt_lambda
    
    #Inner k-fold for finding optimal nbrs
    (opt_val_err_knn,
        opt_nbrs,
        test_err_vs_nbrs) = knearest_validate(X_train, y_train, nbrs_range, internal_cross_validation)
    opt_ns[k] = opt_nbrs

    #Baseline prediction
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
    
    #Logistic Regression
    model = lm.LogisticRegression(penalty='l2', C=(1/opt_lambda), solver='lbfgs', max_iter=1000)
    model.fit(X_train_rlogr, y_train)
            
    y_est_rlogr = model.predict(X_test_rlogr)

    y_true_logr.extend(y_test)
    y_pred_logr.extend(y_est_rlogr)

    rlogr_error = 1 - accuracy_score(y_test, y_est_rlogr)
    Error_test_logr[k] = rlogr_error

    #KNN
    knclassifier = KNeighborsClassifier(opt_nbrs)
    knclassifier.fit(X_train, y_train)
    y_est_knn = knclassifier.predict(X_test)

    y_true_knn.extend(y_test)
    y_pred_knn.extend(y_est_knn)

    error = 1 - accuracy_score(y_test, y_est_knn)
    Error_test_knn[k] = error
    print(test_err_vs_nbrs)
    k = k + 1

df_results = pd.DataFrame({
    'fold': range(1, K+1),
    'opt_n': opt_ns.ravel(),
    'KNN_test_error': Error_test_knn.ravel(),
    'opt_lambda': opt_lambdas.ravel(),
    'RLOGR_test_error': Error_test_logr.ravel(),
    'baseline_test_error': Error_test_baseline.ravel()
})

print(df_results)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    # Labels specific to your dataset
    labels = ['Class A', 'Class B']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotated labels: count + percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    # Plot
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=False,
                     xticklabels=labels, yticklabels=labels, linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save to file if specified
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax

# Plot KNN confusion matrix with improved visualization
plot_confusion_matrix(
    y_true_knn, 
    y_pred_knn, 
    f'KNN Confusion Matrix',
    save_path='knn_confusion_matrix_enhanced.png'
)

plot_confusion_matrix(
    y_true_logr, 
    y_pred_logr, 
    f'Logistic Regression Confusion Matrix',
    save_path='logr_confusion_matrix.png'
)

plt.show()