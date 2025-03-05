import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns

# import data and load it in X
def import_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "Concrete_Data.xls"
    df = pd.read_excel(data_path)
    add_grade_column(df)
    attributeNames = df.columns.values

    classLabels = df["Grade"].to_list()
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames, range(4)))
    
    y = np.asarray([classDict[value] for value in classLabels])

    X = np.empty((1030, 8))
    for i, col_id in enumerate(range(0, 8)):
        X[:, i] = df.iloc[0:1030, col_id].values
    
    y2 = df.iloc[0:1030,8].values

    N = len(y)
    M = len(attributeNames)
    C = len(classNames) 

    return X, y, N, M, C, classNames, attributeNames, y2

def scatter(X, C, y, classNames, attributeNames):
    i = 1
    j = 6

    f = plt.figure()
    plt.title("Concrete data")

    for c in range(C):
        class_mask = y == c 
        plt.plot(X[class_mask, i], X[class_mask, j], "o", alpha=0.3)

    plt.legend(classNames)
    plt.xlabel(attributeNames[i])
    plt.ylabel(attributeNames[j])
    plt.show()

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
    plt.show()

def PCA_attribute(X,attributeNames,N,y):
    Y = X - np.ones((N, 1)) * X.mean(0)
    Y = Y * (1 / np.std(Y, 0))
    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    N, M = X.shape

    pcs = [0,1,2,3,4,5,6,7]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharey = True)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(pcs):
            pc = pcs[i]
            ax.bar(np.arange(M), V[:, pc])
            ax.set_xticks(np.arange(M))
            ax.set_title(f"Principal Component {pc + 1}")
            ax.set_xlabel("Attributes")
            ax.set_ylabel("Loading Coefficient")
            ax.grid(True)
        else:
            ax.axis("off")
            
    fig.tight_layout()
    plt.show()

def PCA_scatter(X, N, C, classNames,y):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    Y = Y / np.std(Y, axis=0)
    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T

    Z = Y @ V
    
    # Define the pairs of PCs to plot
    scatter_pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax, (i, j) in zip(axes, scatter_pairs):
        for c in range(C):
            class_mask = (y == c)
            ax.plot(Z[class_mask, i], Z[class_mask, j], "o", alpha=0.5, label=classNames[c])
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        ax.set_title(f"PC{i+1} vs PC{j+1}")
        ax.grid(True)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=C, bbox_to_anchor=(0.5, 0.05))
    
    fig.suptitle("Concrete data: PCA", fontsize=16)
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()

def print_summary_statistics(data):
    summary_stats = data.describe()
    for column in summary_stats.columns:
        print(f"Summary statistics for {column}:")
        print(summary_stats[column])
        print()

def assign_grade(value):
    if value < 20:
        return 'Low Strength Concrete'
    elif 20 <= value < 50:
        return 'Moderate Strength Concrete '
    else:
        return 'High Strength Concrete'

def add_grade_column(df):
    df['Grade'] = df[df.columns[8]].apply(assign_grade)
    return df

def correlation(X,y,attributeName):
    X = pd.DataFrame(X)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10)) 
    axes = axes.flatten() 

    sns.set_style("whitegrid")

    for i, column in enumerate(X.columns):
        axes[i].scatter(X[column], y, alpha=0.6, color='royalblue', edgecolors='black')
        xlabel = "Age (days)" if i == 7 else f"Compound {i+1} (kg/m^3)"
        axes[i].set_xlabel(xlabel, fontsize=12, labelpad=10)
        axes[i].set_ylabel("Concrete Compressive Strength (MPa)", fontsize=12, labelpad=10)
        axes[i].grid(True, linestyle='--', alpha=0.6)  # Add subtle grid lines

    fig.suptitle("Scatter Plots of Attributes vs Concrete Compressive Strength", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()

def main():
    X, y, N, M, C, classNames, attributeNames, y2 = import_data()
    #scatter(X,C,y,classNames, attributeNames)
    #PCA_variance(X,N)
    #PCA_attribute(X,attributeNames,N,y)
    #PCA_scatter(X,N,C,classNames,y)
    correlation(X,y2,attributeNames)

if __name__ == "__main__":
    main()