# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import torch
from load_data import X_normalized, y2
from dtuimldmtools import rlr_validate, train_neural_net, visualize_decision_boundary
from scipy import stats

# %%
# Configuration
K1 = 10  # Outer folds
K2 = 10  # Inner folds
random_state = 42

# Define hyperparameter ranges
lambdas = np.logspace(-2, 7, 30)  # Range for Ridge regression
hidden_units = [1, 2, 5, 10, 20]  # Range for ANN (including h=1 as required)

# Add offset attribute for linear models
X_with_offset = np.concatenate((np.ones((X_normalized.shape[0], 1)), X_normalized), 1)

# Initialize arrays to store performance metrics
baseline_errors = np.empty(K1)
ridge_errors = np.empty(K1)
ann_errors = np.empty(K1)

# Store optimal parameters
optimal_lambdas = np.empty(K1)
optimal_hidden_units = np.empty(K1)

# %%
print("Starting two-level cross-validation...")

# Set torch seed for reproducibility
torch.manual_seed(random_state)

# Outer CV
outer_cv = model_selection.KFold(n_splits=K1, shuffle=True, random_state=random_state)

k = 0
for train_index, test_index in outer_cv.split(X_normalized):
    print(f"\nOuter fold {k+1}/{K1}")
    
    # Split data for this outer fold
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    X_train_offset, X_test_offset = X_with_offset[train_index], X_with_offset[test_index]
    y_train, y_test = y2[train_index], y2[test_index]
    
    # ---------- MODEL 1: BASELINE MODEL ----------
    y_mean = np.mean(y_train)
    y_pred_baseline = np.full(y_test.shape, y_mean)
    baseline_errors[k] = np.mean((y_test - y_pred_baseline)**2)
    
    print(f"Baseline MSE: {baseline_errors[k]:.4f}")
    
    # ---------- MODEL 2: RIDGE REGRESSION ----------
    # Find optimal lambda using inner cross-validation
    opt_val_err, opt_lambda, _, _, _ = rlr_validate(X_train_offset, y_train, lambdas, K2)
    optimal_lambdas[k] = opt_lambda
    
    # Train final Ridge model with optimal lambda on entire training set
    Xty = X_train_offset.T @ y_train
    XtX = X_train_offset.T @ X_train_offset
    
    # Set regularization matrix with optimal lambda
    lambdaI = opt_lambda * np.eye(X_train_offset.shape[1])
    lambdaI[0, 0] = 0  # Do not regularize bias term
    
    # Calculate weights
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    
    # Predict on test set and calculate error
    y_pred_ridge = X_test_offset @ w_rlr
    ridge_errors[k] = np.mean((y_test - y_pred_ridge)**2)
    
    print(f"Ridge MSE: {ridge_errors[k]:.4f} (λ = {opt_lambda:.2e})")
    
    # ---------- MODEL 3: ARTIFICIAL NEURAL NETWORK ----------
    # Initialize storage for inner fold results
    ann_inner_errors = np.zeros(len(hidden_units))
    
    # Inner CV to find best number of hidden units
    inner_cv = model_selection.KFold(n_splits=K2, shuffle=True, random_state=random_state)
    
    for i, h in enumerate(hidden_units):
        fold_errors = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]
            
            # Convert to PyTorch tensors
            X_torch = torch.tensor(X_inner_train, dtype=torch.float)
            y_torch = torch.tensor(y_inner_train, dtype=torch.float).reshape(-1, 1)
            X_val_torch = torch.tensor(X_inner_val, dtype=torch.float)
            y_val_torch = torch.tensor(y_inner_val, dtype=torch.float).reshape(-1, 1)
            
            # Define model architecture with h hidden units
            M = X_torch.shape[1]  # Number of features
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h),  # M features to h hidden units
                torch.nn.ReLU(),        # ReLU activation
                torch.nn.Linear(h, 1),  # h hidden units to 1 output
                # No activation for regression
            )
            
            # Define loss function
            loss_fn = torch.nn.MSELoss()
            
            # Train the network
            net, final_loss, _ = train_neural_net(model, 
                                               loss_fn, 
                                               X=X_torch, 
                                               y=y_torch,
                                               n_replicates=1,  # Number of times to train
                                               max_iter=5000)   # Max iterations
            
            # Evaluate on validation set
            net.eval()  # Set model to evaluation mode
            with torch.no_grad():  # No need to compute gradients for evaluation
                y_val_est = net(X_val_torch)
                val_error = torch.nn.MSELoss()(y_val_est, y_val_torch).item()
            
            fold_errors.append(val_error)
        
        # Average error for this number of hidden units
        ann_inner_errors[i] = np.mean(fold_errors)
        print(f"  Inner CV: {h} hidden units - MSE: {ann_inner_errors[i]:.4f}")
    
    # Select optimal number of hidden units
    opt_h_idx = np.argmin(ann_inner_errors)
    opt_h = hidden_units[opt_h_idx]
    optimal_hidden_units[k] = opt_h
    print(f"  Selected optimal hidden units: {opt_h}")
    
    print("test 1")
    # Convert entire training and test sets to PyTorch tensors
    X_torch_train = torch.tensor(X_train, dtype=torch.float)
    print("test 2")
    y_torch_train = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)
    print("test 3")
    X_torch_test = torch.tensor(X_test, dtype=torch.float)
    print("test 4")
    # Define final model architecture with optimal hidden units
    M = X_torch_train.shape[1]
    final_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, int(opt_h)),
        torch.nn.ReLU(),
        torch.nn.Linear(int(opt_h), 1),
    )
    
    # Train final ANN model with optimal hidden units on entire training set
    final_net, _, _ = train_neural_net(final_model, 
                                    torch.nn.MSELoss(), 
                                    X=X_torch_train, 
                                    y=y_torch_train,
                                    n_replicates=3,  # Train 3 networks and select best
                                    max_iter=10000)
    
    # Predict on test set and calculate error
    final_net.eval()
    with torch.no_grad():
        y_pred_ann_torch = final_net(X_torch_test)
        y_pred_ann = y_pred_ann_torch.numpy().flatten()
    
    ann_errors[k] = np.mean((y_test - y_pred_ann)**2)
    
    print(f"ANN MSE: {ann_errors[k]:.4f} (hidden units = {opt_h})")
    
    k += 1

# %%
# Statistical comparison and visualization
print("\n--- FINAL RESULTS ---")
print(f"Baseline - Mean MSE: {np.mean(baseline_errors):.4f} ± {np.std(baseline_errors):.4f}")
print(f"Ridge   - Mean MSE: {np.mean(ridge_errors):.4f} ± {np.std(ridge_errors):.4f}")
print(f"ANN     - Mean MSE: {np.mean(ann_errors):.4f} ± {np.std(ann_errors):.4f}")

print("\nAverage optimal parameters:")
print(f"Ridge - λ: {np.mean(optimal_lambdas):.2e}")
print(f"ANN - Hidden units: {np.mean(optimal_hidden_units):.1f}")

# Paired t-tests
alpha = 0.05
print("\nStatistical tests (paired t-test, α=0.05):")
t_ridge_vs_baseline, p_ridge_vs_baseline = stats.ttest_rel(ridge_errors, baseline_errors)
t_ann_vs_baseline, p_ann_vs_baseline = stats.ttest_rel(ann_errors, baseline_errors)
t_ridge_vs_ann, p_ridge_vs_ann = stats.ttest_rel(ridge_errors, ann_errors)

print(f"Ridge vs Baseline: p={p_ridge_vs_baseline:.4f} ({'Significant' if p_ridge_vs_baseline < alpha else 'Not significant'})")
print(f"ANN vs Baseline: p={p_ann_vs_baseline:.4f} ({'Significant' if p_ann_vs_baseline < alpha else 'Not significant'})")
print(f"Ridge vs ANN: p={p_ridge_vs_ann:.4f} ({'Significant' if p_ridge_vs_ann < alpha else 'Not significant'})")

# %%
# Visualize results
plt.figure(figsize=(14, 6))

# Box plot of errors
plt.subplot(1, 2, 1)
plt.boxplot([baseline_errors, ridge_errors, ann_errors])
plt.xticks([1, 2, 3], ['Baseline', 'Ridge', 'ANN'])
plt.ylabel('Mean Squared Error')
plt.title('Model Performance Comparison')
plt.grid(axis='y')

# Bar plot of average errors with error bars
plt.subplot(1, 2, 2)
models = ['Baseline', 'Ridge', 'ANN']
mean_errors = [np.mean(baseline_errors), np.mean(ridge_errors), np.mean(ann_errors)]
std_errors = [np.std(baseline_errors), np.std(ridge_errors), np.std(ann_errors)]

plt.bar(models, mean_errors, yerr=std_errors, capsize=10)
plt.ylabel('Mean Squared Error')
plt.title('Average Model Performance with Standard Deviation')
plt.grid(axis='y')

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
plt.show()
# %%