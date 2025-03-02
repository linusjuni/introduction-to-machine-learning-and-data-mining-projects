import matplotlib.pyplot as plt
import numpy as np
import dtuimldmtools
import pandas as pd
import os
from scipy.linalg import svd
from pathlib import Path

project_root = Path(__file__).resolve()
while not (project_root / ".git").exists() and project_root != project_root.parent:
    project_root = project_root.parent
filename = project_root / "data" / "Concrete_Data.xls"
data = pd.read_excel(filename)
X = data.values

print(X[:5, :])
def extract_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(project_root, "data", "Concrete_Data.xls")
    data = pd.read_excel(filename)
    attributesNames = data.columns.values
    return data, attributesNames

def boxplots(df,attr):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, column in enumerate(attr):
        df.boxplot(column=column, ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')
    
    plt.tight_layout()
    plt.show()

def assign_grade(value):
    if value < 20:
        return 'Low Strength Concrete'
    elif 20 <= value < 50:
        return 'Moderate Strength Concrete '
    elif 50 <= value < 150:
        return 'High Strength Concrete'
    else:
        return 'Ultra High Strength Concrete'

def plots_strength(df,attr):
    df[attr[-1]].hist()
    #plt.show()
    #df.boxplot(attr[-1])
    quantiles = df[attr[-1]].quantile([0.0,0.25, 0.5, 0.75,1.0])
    middle = ((quantiles[1.0] - quantiles[0.0]) / 2) + quantiles[0.0]
    first = ((middle - quantiles[0.0]) / 2) + quantiles[0.0]
    third = ((quantiles[1.0] - middle) / 2) + middle
    grades = [first, middle, third]
    print(grades)
    return grades

def hists(df,attr):
    n_cols = 3
    n_rows = (len(attr) // n_cols) + (1 if len(attr) % n_cols != 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()
    for i, column in enumerate(attr):
        axes[i].hist(df[column], bins=20, edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout() 
    plt.show()

def add_grade_column(df):
    df['Grade'] = df[df.columns[8]].apply(assign_grade)
    print(df)

def main():
    df, attr = extract_data()
    #plots_strength(df,attr)
    #boxplots(df,attr)
    #add_grade_column(df)
    hists(df,attr)

if __name__ == "__main__":
    main()