# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from load_data import X_normalized, y2  # X_normalized is already standardized
from dtuimldmtools import rlr_validate

# %%
# Add offset attribute (column of ones)
X = np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), 1)
N, M = X.shape

# Create cross-validation partition for evaluation
K = 10  # Number of folds
CV = model_selection.KFold(K, shuffle=True, random_state=42)

# Values of lambda - refined range focusing more on smaller values where we expect to see improvement
lambdas = np.logspace(-2, 7, 50)

# Initialize variables
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
Error_train_nofeatures = np.empty((K, 1))
Error_test_nofeatures = np.empty((K, 1))
w_rlr = np.empty((M, K))
w_noreg = np.empty((M, K))

# Store lambda values and corresponding validation errors
lambda_error = []

# Additional arrays to track lambda performance across all folds
all_train_err_vs_lambda = np.zeros((len(lambdas), K))
all_test_err_vs_lambda = np.zeros((len(lambdas), K))
all_mean_w_vs_lambda = []

# %%
k = 0
for train_index, test_index in CV.split(X, y2):
    # Extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y2[train_index]
    X_test = X[test_index]
    y_test = y2[test_index]
    
    internal_cross_validation = 10
    
    # Find optimal lambda using internal cross-validation
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    # Store all errors for each lambda value to compute average across folds
    all_train_err_vs_lambda[:, k] = train_err_vs_lambda
    all_test_err_vs_lambda[:, k] = test_err_vs_lambda
    all_mean_w_vs_lambda.append(mean_w_vs_lambda)
    
    # Store lambda and error values for plotting
    lambda_error.append((opt_lambda, opt_val_err))
    
    # Matrix calculations
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Baseline: Mean prediction
    Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]
    
    # Ridge regression with optimal lambda
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do not regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum() / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum() / y_test.shape[0]
    
    # Unregularized linear regression
    w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
    Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum() / y_train.shape[0]
    Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum() / y_test.shape[0]
    
    # Print debug info for this fold
    print(f"Fold {k+1}: Optimal λ = {opt_lambda:.2e}, Validation error = {opt_val_err:.4f}")
    
    # Specific fold-level diagnostics
    print(f"  Lambda range analysis (Fold {k+1}):")
    print(f"  - Min test error: {np.min(test_err_vs_lambda):.4f} at λ = {lambdas[np.argmin(test_err_vs_lambda)]:.2e}")
    print(f"  - Max test error: {np.max(test_err_vs_lambda):.4f} at λ = {lambdas[np.argmax(test_err_vs_lambda)]:.2e}")
    print(f"  - Error range: {np.max(test_err_vs_lambda) - np.min(test_err_vs_lambda):.4f}")
    
    k += 1

# %%
# Calculate average errors across all folds for each lambda
avg_train_err = np.mean(all_train_err_vs_lambda, axis=1)
avg_test_err = np.mean(all_test_err_vs_lambda, axis=1)

# Identify the best lambda based on average test error across all folds
best_lambda_idx = np.argmin(avg_test_err)
best_lambda = lambdas[best_lambda_idx]

# Collect all validation errors across folds
all_lambdas = [x[0] for x in lambda_error]
all_errors = [x[1] for x in lambda_error]

# Calculate mean optimal lambda
mean_opt_lambda = np.mean(all_lambdas)

# Calculate the standard deviation of errors across folds (for error bars)
std_test_err = np.std(all_test_err_vs_lambda, axis=1)

# %%
# Enhanced visualization of generalization error vs lambda
plt.figure(figsize=(15, 10))

# Plot 1: Main generalization error vs lambda plot with error bars
plt.subplot(2, 2, 1)
plt.errorbar(lambdas, avg_test_err, yerr=std_test_err, fmt='r.-', capsize=3, label='Validation Error (with std dev)')
plt.axvline(x=best_lambda, color='g', linestyle='--', label=f'Best λ = {best_lambda:.2e}')
plt.xscale('log')
plt.xlabel('Regularization factor (λ)')
plt.ylabel('Mean Squared Error')
plt.title('Generalization Error vs Lambda (Averaged Across All Folds)')
plt.grid(True)
plt.legend()

# Plot 2: Compare training and test errors
plt.subplot(2, 2, 2)
plt.semilogx(lambdas, avg_train_err, 'b.-', label='Training Error')
plt.semilogx(lambdas, avg_test_err, 'r.-', label='Validation Error')
plt.axvline(x=best_lambda, color='g', linestyle='--', label=f'Best λ = {best_lambda:.2e}')
plt.xlabel('Regularization factor (λ)')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Validation Error Across Lambda Values')
plt.legend()
plt.grid(True)

# Plot 3: Zoomed in view around the optimal lambda region
zoom_start = max(0, best_lambda_idx - 10)
zoom_end = min(len(lambdas), best_lambda_idx + 4)
plt.subplot(2, 2, 3)
plt.plot(lambdas[zoom_start:zoom_end], avg_test_err[zoom_start:zoom_end], 'ro-')
plt.axvline(x=best_lambda, color='g', linestyle='--', label=f'Best λ = {best_lambda:.2e}')
plt.xlabel('Regularization factor (λ)')
plt.ylabel('Validation Error')
plt.title('Zoomed View Around Optimal Lambda Region')
plt.grid(True)
plt.legend()

# Plot 4: Feature weights vs lambda (average across folds)
avg_mean_w_vs_lambda = np.mean(np.array(all_mean_w_vs_lambda), axis=0)
plt.subplot(2, 2, 4)
plt.semilogx(lambdas, avg_mean_w_vs_lambda.T[:, 1:], '.-')
plt.xlabel('Regularization factor (λ)')
plt.ylabel('Mean Coefficient Values')
plt.title('Effect of λ on Model Coefficients (Avg Across Folds)')
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Display results
print(f"\nResults from {K}-fold cross-validation:")
print(f"Mean optimal λ from individual fold optimizations: {mean_opt_lambda:.4e}")
print(f"Best λ based on average validation error across all folds: {best_lambda:.4e}")
print(f"Minimum average validation error: {np.min(avg_test_err):.4f}")
print(f"Maximum average validation error: {np.max(avg_test_err):.4f}")
print(f"Difference between max and min error: {np.max(avg_test_err) - np.min(avg_test_err):.4f}")

print("\nModel Comparison:")
print("-------------------------------------")
print("Baseline (mean prediction):")
print(f"- Test error: {Error_test_nofeatures.mean():.4f}")

print("\nLinear regression (no regularization):")
print(f"- Training error: {Error_train.mean():.4f}")
print(f"- Test error: {Error_test.mean():.4f}")
print(f"- R² test: {(Error_test_nofeatures.mean() - Error_test.mean()) / Error_test_nofeatures.mean():.4f}")

print("\nRegularized linear regression (Ridge):")
print(f"- Training error: {Error_train_rlr.mean():.4f}")
print(f"- Test error: {Error_test_rlr.mean():.4f}")
print(f"- R² test: {(Error_test_nofeatures.mean() - Error_test_rlr.mean()) / Error_test_nofeatures.mean():.4f}")

# %%
# Plot feature weights for regularized model
plt.figure(figsize=(12, 6))
feature_names = ['Offset', 'Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
mean_weights = np.mean(w_rlr, axis=1)
plt.barh(range(len(mean_weights)), mean_weights)
plt.yticks(range(len(mean_weights)), feature_names)
plt.xlabel('Coefficient Value')
plt.title('Average Feature Weights in Ridge Regression')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
# %%

# Additional diagnostic plot to compare test error ranges between lambda values
plt.figure(figsize=(10, 6))
plt.boxplot([all_test_err_vs_lambda[i, :] for i in range(0, len(lambdas), len(lambdas)//10)])
lambda_labels = [f"{lambdas[i]:.2e}" for i in range(0, len(lambdas), len(lambdas)//10)]
plt.xticks(range(1, len(lambda_labels) + 1), lambda_labels, rotation=45)
plt.xlabel('Lambda Values (subset)')
plt.ylabel('Validation Error')
plt.title('Distribution of Validation Errors Across Different Lambda Values')
plt.tight_layout()
plt.show()

print("LOL")
# %%
