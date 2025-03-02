import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dtuimldmtools
import os
from scipy.linalg import svd
from pathlib import Path

# import data and load it in X
data_path = Path(__file__).resolve().parent.parent / "data" / "Concrete_Data.xls"
df = pd.read_excel(data_path)
X = df.values

summary_stats = df.describe()
for column in summary_stats:
    print(f"Summary statistics for {column}:")
    print(summary_stats[column])
    print()