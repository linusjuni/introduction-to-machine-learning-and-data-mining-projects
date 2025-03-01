import matplotlib.pyplot as plt
import numpy as np
import dtuimldmtools
import pandas as pd
import os
from scipy.linalg import svd

def extract_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(project_root, "data", "Concrete_Data.xls")
    data = pd.read_excel(filename)
    attributesNames = data.columns.values
    return data, attributesNames

def boxplots(df,attr):
    # Set up a grid with 3 rows and 3 columns (since you have 9 attributes)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the axes array to easily iterate over
    
    # Loop through the attributes and corresponding axes
    for i, column in enumerate(attr):
        # Plot each boxplot on a different subplot
        df.boxplot(column=column, ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')  # Set title for each plot
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def assign_grade(value):
    if value < 22.4:
        return 'Grade 1'
    elif 22.4 <= value < 42.5:
        return 'Grade 2'
    elif 42.5 <= value < 62.5:
        return 'Grade 3'
    else:
        return 'Grade 4'

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
    # Set up the number of rows and columns for the grid layout
    n_cols = 3  # You can adjust this based on your needs
    n_rows = (len(attr) // n_cols) + (1 if len(attr) % n_cols != 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as needed
    axes = axes.flatten()  # Flatten axes for easier iteration

    # Loop through the attributes and create a histogram for each
    for i, column in enumerate(attr):
        axes[i].hist(df[column], bins=20, edgecolor='black')  # Customize number of bins as needed
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)  # Add grid for easier readability
    
    # Turn off unused axes (if the number of attributes isn't a perfect grid)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()  # Adjust layout to prevent overlapping
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