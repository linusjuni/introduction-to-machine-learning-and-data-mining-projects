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