import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from all_cnn_bi_skippy_cut import inv_gray, Skip_manager,CNN_conf
from keras.utils import plot_model

import sys

from mipego.mipego import Solution
from mipego.Bi_Objective import *
import math

if len(sys.argv) != 4 and len(sys.argv) != 6:
    print("usage: python3 load_data.py 'data_file_name.json' init_solution_number zoom(0,1) (optional: ref_time ref_loss)")
    exit(0)
file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

init_amount = int(sys.argv[2])
zoom = int(sys.argv[3])

if len(sys.argv) == 6:
    ref_time = float(sys.argv[4])
    ref_loss = float(sys.argv[5])
else:
    ref_time = None
    ref_loss = None

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

#calculate percentage of disfunctional networks (too large to train on tritanium gpu)
disfunctional = 0
total = 0
for i in range(len(solutions)):
    total +=1
    if solutions[i].time >= 80000.0 - 1.0:
        disfunctional +=1

print("Percentage disfunctional networks: " + str(disfunctional * 100 / total) + "%")
pauser = 0.008
if False:
    if len(surr_time_fit_hist) > 0:
        print('surr_time_fit_hist:')
        print(surr_time_fit_hist)
    else:
        print('No surr_time_fit_hist')
    if len(surr_time_mies_hist) > 0:
        print('surr_time_mies_hist:')
        print(surr_time_mies_hist)
    else:
        print('No surr_time_mies_hist')
    if len(surr_loss_fit_hist) > 0:
        print('surr_loss_fit_hist:')
        print(surr_loss_fit_hist)
    else:
        print('No surr_loss_fit_hist')
    if len(surr_loss_mies_hist) > 0:
        print('surr_loss_mies_hist:')
        print(surr_loss_mies_hist)
    else:
        print('No surr_loss_fit_hist')

    if len(time_between_gpu_hist) > 0:
        print('time_between_gpu_hist:')
        print(time_between_gpu_hist)
    else:
        print('no time_between_gpu_hist')

time = [x.time for x in solutions]
loss = [x.loss for x in solutions]

#print('time:')
#print(time)
#print('loss:')
#print(loss)
x_bound = min(0.0,min(time)),max(time)
y_bound = min(0.0,min(loss)),max(loss)

plt.ion()
for i in range(1,0):#len(solutions)):
    plt.clf()
    plt.xlabel('time (s)')
    plt.ylabel('loss')
    axes = plt.gca()
    axes.set_xlim([x_bound[0],x_bound[1]])
    axes.set_ylim([y_bound[0],y_bound[1]])

    par = pareto(solutions[0:i])

    par_time = [x.time for x in par]
    par_loss = [x.loss for x in par]

    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig, ax = plt.subplots()
    plt.plot(time[0:min(i-1,init_amount)], loss[0:min(i-1,init_amount)],'yo')
    if i-1 > init_amount:
        plt.plot(time[init_amount:i-1], loss[init_amount:i-1], 'o')
    plt.plot(par_time, par_loss, 'ro')
    plt.plot(time[i], loss[i], 'go')
    #plt.set_title('sms-mip-ego')
    #plt.show()
    plt.pause(pauser)
par = pareto(solutions)
quicksort_par(par,0,len(par)-1)
par_time = [x.time for x in par]
par_loss = [x.loss for x in par]
HV = hyper_vol(par, solutions, ref_time, ref_loss)

print("Hyper Volume:")
print(HV)
print("len pareto front:")
print(len(par))
print("len loss r2 score:")
print(len(all_loss_r2))
print("len time r2 score:")
print(len(all_time_r2))
print("paretofront:")
for i in range(len(par)):
    print("time: " + str(par[i].time) + ", loss: " + str(par[i].loss) + ", acc: " + str(np.exp(-par[i].loss)))
if len(all_time_r2) > 0 and len(all_loss_r2) > 0:
    print("all_time_r2 average:")
    print(np.average(np.array(all_time_r2)))
    print("all_loss_r2 average:")
    print(np.average(np.array(all_loss_r2)))
#print(par[0].var_name)
print(par)
for i in range(len(par)):
    print(par[i].to_dict())
    #model = CNN_conf(par[i].to_dict(),test=True)
    #plot_model(model, to_file='conf_pareto_skippy_' + str(i)+ '.png',show_shapes=True,show_layer_names=True)

print('top 7 highest accuracy:')
top_seven = np.argsort(loss)[0:7]
for i in top_seven:
    print("time: " + str(solutions[i].time) + ", loss: " + str(solutions[i].loss) + ", acc: " + str(np.exp(-solutions[i].loss)))
print()
for i in top_seven:
    print(solutions[i].to_dict())
    print()

#sorter = np.argsort([x.time for x in solutions])
#for i in range(len(sorter)):
#    print(solutions[sorter[i]].to_dict())
#    model = CNN_conf(solutions[sorter[i]].to_dict(),test=True)
#    plot_model(model, to_file='conf_solution_skippy_' + str(i)+ '.png',show_shapes=True,show_layer_names=True)
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
plt.xlabel('time (s)')#CHRIS x^2
plt.ylabel('loss')#(x-2)^2
axes = plt.gca()
axes.set_xlim([x_bound[0],x_bound[1]])
axes.set_ylim([y_bound[0],y_bound[1]])
#plt.plot(time[0:init_amount], loss[0:init_amount],'yo')
#plt.plot(time[init_amount:], loss[init_amount:], 'o')
#plt.plot(par_time, par_loss, 'ro')
#plt.pause(float('inf'))
init = pd.DataFrame(data={'time':time[0:init_amount], 'loss':loss[0:init_amount]})
a=sns.scatterplot(x='time', y='loss', data=init,color = 'y',label='Init',marker='+',s=100)
a.set_xscale('log')
heur = pd.DataFrame(data={'time':time[init_amount:], 'loss':loss[init_amount:]})
b=sns.scatterplot(x='time', y='loss', data=heur, color = 'b',label='Heuristic',marker='x',s=100)
b.set_xscale('log')
par_data = pd.DataFrame(data={'time':par_time, 'loss':par_loss})
c=sns.scatterplot(x='time', y='loss', data=par_data, color = 'r',label='Pareto',s=100)
c.set_xscale('log')
plt.xlabel('time (s)',size=20)#CHRIS x^2
plt.ylabel('loss',size=20)#(x-2)^2
a.legend()
b.legend()
c.legend()
if zoom:
    a.set(xlim=(0,5000),ylim=(0, 2.5))
    b.set(xlim=(0,5000),ylim=(0, 2.5))
    c.set(xlim=(0,5000),ylim=(0, 2.5))
plt.tight_layout()
plt.pause(float('inf'))
