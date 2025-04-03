import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path
import seaborn as sns

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

    return df, X, y, N, M, C, classNames, attributeNames, y2

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

df, X, y, N, M, C, classNames, attributeNames, y2 = import_data()

print(df.head(5))