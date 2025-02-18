import importlib_resources
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import dtuimldmtools
import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up two levels
filename = os.path.join(project_root, "data", "Concrete_Data.xls")
data = pd.read_excel(filename)
print(data.head())

