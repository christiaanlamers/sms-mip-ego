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
    solutions.append(Solution(x=conf_x,fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

print("len(solutions): " + str(len(solutions)))
print([i.to_dict() for i in solutions])
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
n_points = 1000
for i in name_array[0]:
    print(i)
    x = []
    y = []
    for j in solutions:
        #if j.to_dict()['lr'] < 0.01:
        x.append(j.to_dict()[i])
        y.append(np.exp(-j.loss))
    #print(x)
    try:
        plt.xlabel(i)
        plt.ylabel('accuracy')
        #plt.ylabel('time')
        plt.plot(x,y, 'o')
    except:
        print('could not plot data')
    try:
        spl = UnivariateSpline(x, y)
        max_x = max(x)
        x_model = [k*max_x/n_points for k in range(n_points)]
        y_model = [spl(k) for k in x_model]
        plt.plot(x_model,y_model,color="red")
    except:
        print('could not fit a spline')
    try:
        plt.show()
    except:
        pass
