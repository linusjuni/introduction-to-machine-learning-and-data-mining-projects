import matplotlib.pyplot as plt
import numpy as np
import dtuimldmtools
import pandas as pd
import os
from scipy.linalg import svd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
filename = os.path.join(project_root, "data", "Concrete_Data.xls")
data = pd.read_excel(filename)
print(data.head())

