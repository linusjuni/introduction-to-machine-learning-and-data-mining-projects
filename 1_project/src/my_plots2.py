from import_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_palette("pastel")

sns.set_palette("pastel")

X, y, N, M, C, classNames, attributeNames = import_data()

def PCA_variance(X, N):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    Y = Y * (1 / np.std(Y, 0))
    U, S, Vh = svd(Y, full_matrices=False)

    V = Vh.T

    rho = (S * S) / (S * S).sum()

    threshold = 0.9
    print(rho)
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-")
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
    plt.plot([1, len(rho)], [threshold, threshold], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()
    plt.savefig('/Users/linusjuni/Desktop/variance_pca.png')
    plt.show()

def PCA_attribute(X, attributeNames, N, y):
    Y = X - np.ones((N, 1)) * X.mean(0)
    Y = Y * (1 / np.std(Y, 0))
    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    N, M = X.shape

    pcs = [0, 1, 2, 3, 4, 5, 6, 7]

    fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    
    # Create a bar plot for each selected principal component
    for i, ax in enumerate(axes):
        if i < len(pcs):
            pc = pcs[i]
            # Create a color list for each bar based on its index
            colors = sns.color_palette("pastel", M)
            ax.bar(np.arange(M), V[:, pc], color=colors)  # Assigning different color to each bar
            ax.set_xticks(np.arange(M))
            ax.set_xticklabels(np.arange(1, M + 1))
            ax.set_title(f"Principal Component {pc + 1}")
            ax.set_xlabel("Attributes")
            ax.set_ylabel("Component Coefficients")
            ax.grid(True)
        else:
            ax.axis("off")  # in case there are more subplots than PCs
            
    fig.tight_layout()
    plt.savefig('/Users/linusjuni/Desktop/attributes_pca.png')
    plt.show()

def PCA_scatter(X, N, C, classNames, y):
    Y = X - np.ones((N, 1)) * X.mean(0)
    Y = Y * (1 / np.std(Y, 0))
    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    Z = Y @ V

    i = 0
    j = 1

    f = plt.figure()
    plt.title("Concrete data: PCA")
    for c in range(C):
        class_mask = y == c
        plt.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5, label=classNames[c], color=sns.color_palette("pastel")[c % len(sns.color_palette("pastel"))])
    
    plt.legend()
    plt.xlabel(f"PC{i + 1}")
    plt.ylabel(f"PC{j + 1}")
    plt.savefig('/Users/linusjuni/Desktop/pca_scatter.png')
    plt.show()

PCA_variance(X,N)
PCA_attribute(X,attributeNames,N,y)
PCA_scatter(X, N, C, classNames, y)