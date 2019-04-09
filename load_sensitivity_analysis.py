import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from all_cnn_bi_skippy import inv_gray, Skip_manager,CNN_conf
from keras.utils import plot_model
import sys
from mipego.mipego import Solution
from mipego.Bi_Objective import *
import math
from scipy.interpolate import UnivariateSpline
import numbers
from pandas.plotting import parallel_coordinates
import sklearn
from sklearn.cluster import KMeans, DBSCAN
import sklearn.metrics as sm
from apyori import apriori
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami

problem = {
    'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]]*3
}

# Generate samples
param_values = saltelli.sample(problem, 1000)

# Run model (example)
Y = Ishigami.evaluate(param_values)

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=True)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)
