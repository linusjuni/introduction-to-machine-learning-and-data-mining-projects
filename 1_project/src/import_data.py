import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path

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

    N = len(y)
    M = len(attributeNames)
    C = len(classNames) 

    return X, y, N, M, C, classNames, attributeNames

def scatter(X, C, y, classNames, attributeNames):
    i = 1
    j = 6

    f = plt.figure()
    plt.title("Concrete data")

    for c in range(C):
        # select indices belonging to class c:
        class_mask = y == c 
        plt.plot(X[class_mask, i], X[class_mask, j], "o", alpha=0.3)

    plt.legend(classNames)
    plt.xlabel(attributeNames[i])
    plt.ylabel(attributeNames[j])
    plt.show()

def PCA_variance(X, N):
    Y = X - np.ones((N, 1)) * X.mean(axis=0)

    U, S, Vh = svd(Y, full_matrices=False)

    V = Vh.T

    rho = (S * S) / (S * S).sum()

    threshold = 0.9

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
    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    N, M = X.shape

    # We saw in 2.1.3 that the first 3 components explaiend more than 90
    # percent of the variance. Let's look at their coefficients:
    pcs = [ 1, 2, 3, 4, 5]
    legendStrs = ["PC" + str(e + 1) for e in pcs]
    bw = 0.2
    r = np.arange(1, M + 1)

    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)

    #plt.xticks(r + bw, attributeNames)
    plt.xlabel("Attributes")
    plt.ylabel("Component coefficients")
    plt.legend(legendStrs)
    plt.grid()
    plt.title("NanoNose: PCA Component Coefficients")
    plt.show()

    # Inspecting the plot, we see that the 2nd principal component has large
    # (in magnitude) coefficients for attributes A, E and H. We can confirm
    # this by looking at it's numerical values directly, too:
    print("PC2:")
    print(V[:, 1].T)

    # How does this translate to the actual data and its projections?
    # Looking at the data for water:

    # Projection of water class onto the 2nd principal component.
    all_water_data = Y[y == 4, :]

    print("First water observation")
    print(all_water_data[0, :])

    # Based on the coefficients and the attribute values for the observation
    # displayed, would you expect the projection onto PC2 to be positive or
    # negative - why? Consider *both* the magnitude and sign of *both* the
    # coefficient and the attribute!

    # You can determine the projection by (remove comments):
    # print("...and its projection onto PC2") 
    # print(all_water_data[0, :] @ V[:, 1])  


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
    elif 50 <= value < 150:
        return 'High Strength Concrete'
    else:
        return 'Ultra High Strength Concrete'

def add_grade_column(df):
    df['Grade'] = df[df.columns[8]].apply(assign_grade)
    return df
"""
def main():
    X, y, N, M, C, classNames, attributeNames = import_data()
    #scatter(X,C,y,classNames, attributeNames)
    PCA_variance(X,N)
    #PCA_attribute(X,attributeNames,N,y)
    #print(data.head(10))
    #print(data)
    #print(values)
    #print(headers)

if __name__ == "__main__":
    main()
"""