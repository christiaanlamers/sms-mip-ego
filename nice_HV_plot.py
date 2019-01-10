import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


import sys

from mipego.mipego import Solution
from mipego.Bi_Objective import *

zoom = True

par_time = [0.1,0.3,0.4,0.7]
par_loss = [0.8,0.6,0.3,0.1]

#if all_time_r2 is not None and all_loss_r2 is not None:
#    print("all_time_r2:")
#    print(all_time_r2)
#    print("all_loss_r2:")
#    print(all_loss_r2)
#print(par)
#print(par[0].var_name)
#for i in range(1):#range(len(par)):
#    vec = par[i].to_dict()
#    print(vec)
#    print(objective(vec))
plt.clf()
#plt.xlabel('time')
#plt.ylabel('loss')
#plt.plot(time[0:init_amount], loss[0:init_amount],'yo')
#plt.plot(time[init_amount:], loss[init_amount:], 'o')
#plt.plot(par_time, par_loss, 'ro')
#plt.pause(float('inf'))

ref_point_values = (1.0,1.0)

#plot first box
plt.plot([par_time[0], ref_point_values[0]], [ref_point_values[1], ref_point_values[1]], linewidth=2,color='b')
plt.plot([par_time[0], par_time[0]], [ref_point_values[1], par_loss[0]], linewidth=2,color='b')
plt.plot([ref_point_values[0], ref_point_values[0]], [ref_point_values[1], par_loss[0]], linewidth=2,color='b')
#plot all intermediate boxes
for i in range(len(par_time)):
    plt.plot([par_time[i], ref_point_values[0]], [par_loss[i], par_loss[i]], linewidth=2,color='b')
    if i < len(par_time)-1:
        plt.plot([par_time[i+1], par_time[i+1]], [par_loss[i], par_loss[i+1]], linewidth=2,color='b')
        plt.plot([ref_point_values[0], ref_point_values[0]], [par_loss[i], par_loss[i+1]], linewidth=2,color='b')

par_data = pd.DataFrame(data={'objective 1':par_time, 'objective 2':par_loss})
c=sns.scatterplot(x='objective 1', y='objective 2', data=par_data,color=['red','red','red','red'])

ref_point =pd.DataFrame(data={'objective 1':[ref_point_values[0]], 'objective 2':[ref_point_values[1]]})
d=sns.scatterplot(x='objective 1', y='objective 2',data=ref_point,color = 'green')

if zoom:
    c.set(xlim=(0,ref_point_values[0]+0.1),ylim=(0, ref_point_values[1]+0.1))
    d.set(xlim=(0,ref_point_values[0]+0.1),ylim=(0, ref_point_values[1]+0.1))
plt.pause(float('inf'))
