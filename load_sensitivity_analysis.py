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

exit(0)

disfunc_time = 80000 #CHRIS panalty value given to a disfuncitonal network. This differs per experiment

file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

conf_array = data[0]
fit_array = data[1]
time_array = data[2]
loss_array = data[3]
n_eval_array = data[4]
index_array = data[5]
name_array = data[6]

all_time_r2 = []
all_loss_r2 = []
if len(data) > 7:
    all_time_r2 = data[7]
    all_loss_r2 = data[8]

surr_time_fit_hist = []
surr_time_mies_hist = []
surr_loss_fit_hist = []
surr_loss_mies_hist = []
time_between_gpu_hist = []
if len(data) > 9:
    surr_time_fit_hist = data[9]
    surr_time_mies_hist = data[10]
    surr_loss_fit_hist = data[11]
    surr_loss_mies_hist = data[12]
if len(data) > 13:
    time_between_gpu_hist = data[13]


#print(data)
solutions = []
for i in range(len(conf_array)):
    conf_x = [conf_array[i][j] for j in name_array[i]]
    if time_array[i] < disfunc_time:#TODO CHRIS filter out disfunciontal network, note that the time penalty value differs per research instance
        solutions.append(Solution(x=conf_x,fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

print("len(solutions): " + str(len(solutions)))
#print([i.to_dict() for i in solutions])
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
acc_pivot = 0.7
data_lib = {}
data_lib_good = {}
data_lib_bad = {}
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
data_lib['acc'] = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
data_lib['time'] = [i.time for i in solutions]
data_lib_good['acc'] = [np.exp(-i.loss) for i in solutions if np.exp(-i.loss) >= acc_pivot]#[i.time for i in solutions]
data_lib_good['time'] = [i.time for i in solutions if np.exp(-i.loss) >= acc_pivot]
data_lib_bad['acc'] = [np.exp(-i.loss) for i in solutions if np.exp(-i.loss) < acc_pivot]#[i.time for i in solutions]
data_lib_bad['time'] = [i.time for i in solutions if np.exp(-i.loss) < acc_pivot]
#n_points = 1000
stopper = 0
for i in name_array[0]:
    print(i)
    x = []
    x_good = []
    x_bad = []
    #y = []
    for j in solutions:
        #if j.to_dict()['lr'] < 0.01:
        x.append(j.to_dict()[i])
        if np.exp(-j.loss) >= acc_pivot:
            x_good.append(j.to_dict()[i])
        else:
            x_bad.append(j.to_dict()[i])
    #y.append(np.exp(-j.loss))
if isinstance(x[0], numbers.Number): #TODO CHRIS remove this and catch non number related errors
    data_lib[i] = x
    data_lib_good[i] = x_good
    data_lib_bad[i] = x_bad
    elif x[0] == "elu" or x[0] == "relu" or x[0] == "tanh" or x[0] == "sigmoid" or x[0] == "selu":
        elu = []
        relu = []
        tanh= []
        sigmoid = []
        selu = []
        for j in x:
            elu.append(j == "elu")
            relu.append(j=="relu")
            tanh.append(j=="tanh")
            sigmoid.append(j=="sigmoid")
            selu.append(j=="selu")
        data_lib["elu"] = elu
        data_lib["relu"] = relu
        data_lib["tanh"] = tanh
        data_lib["sigmoid"] = sigmoid
        data_lib["selu"] = selu
        
        elu = []
        relu = []
        tanh= []
        sigmoid = []
        selu = []
        for j in x_good:
            elu.append(j == "elu")
            relu.append(j=="relu")
            tanh.append(j=="tanh")
            sigmoid.append(j=="sigmoid")
            selu.append(j=="selu")
data_lib_good["elu"] = elu
data_lib_good["relu"] = relu
data_lib_good["tanh"] = tanh
data_lib_good["sigmoid"] = sigmoid
data_lib_good["selu"] = selu

elu = []
    relu = []
    tanh= []
    sigmoid = []
    selu = []
    for j in x_bad:
        elu.append(j == "elu")
        relu.append(j=="relu")
        tanh.append(j=="tanh")
        sigmoid.append(j=="sigmoid")
        selu.append(j=="selu")
        data_lib_bad["elu"] = elu
        data_lib_bad["relu"] = relu
        data_lib_bad["tanh"] = tanh
        data_lib_bad["sigmoid"] = sigmoid
        data_lib_bad["selu"] = selu
    elif x[0] == "softmax":
        pass
    #softmax = []
    #for j in x:
    #    softmax.append(j=="softmax")
    #data_lib["softmax"] = softmax
    #
    #softmax = []
    #for j in x_good:
    #    softmax.append(j=="softmax")
    #data_lib_good["softmax"] = softmax
    #
    #softmax = []
    #for j in x_bad:
    #    softmax.append(j=="softmax")
    #data_lib_bad["softmax"] = softmax
    else:
        print("error, unknown feature!")
    
    #print(x)


data_panda = pd.DataFrame(data=data_lib)
data_panda_good = pd.DataFrame(data=data_lib_good)
data_panda_bad = pd.DataFrame(data=data_lib_bad)
