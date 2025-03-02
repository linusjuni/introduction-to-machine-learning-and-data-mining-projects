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
    return df, df.values, df.columns.values

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

def main():
    data, values, headers = import_data()
    data = add_grade_column(data)
    print(data)
    print(values)
    print(headers)

if __name__ == "__main__":
    main()