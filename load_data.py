import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys

from mipego.mipego import Solution
from mipego.Bi_Objective import *

if len(sys.argv) != 3 and len(sys.argv) != 5:
    print("usage: python3 load_data.py 'data_file_name.json' init_solution_number (optional: ref_time ref_loss)")
    exit(0)
file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

init_amount = int(sys.argv[2])

if len(sys.argv) == 5:
    ref_time = float(sys.argv[3])
    ref_loss = float(sys.argv[4])
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

#print(data)
solutions = []
for i in range(len(conf_array)):
    solutions.append(Solution(conf_array[i]))
    solutions[i].fit = fit_array[i]
    solutions[i].time = time_array[i]
    solutions[i].loss = loss_array[i]
    solutions[i].n_eval = n_eval_array[i]
    solutions[i].index = index_array[i]
    solutions[i].var_name = name_array[i]

print("len(solutions): " + str(len(solutions)))

pauser = 0.008

time = [x.time for x in solutions]
loss = [x.loss for x in solutions]

#print('time:')
#print(time)
#print('loss:')
#print(loss)
x_bound = min(0.0,min(time)),max(time)
y_bound = min(0.0,min(loss)),max(loss)

plt.ion()
for i in range(1,len(solutions)):
    plt.clf()
    plt.xlabel('time')
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
par = sort_par(par)
HV = hyper_vol(par, solutions, ref_time, ref_loss)
print("Hyper Volume:")
print(HV)
print("paretofront:")
for i in range(len(par)):
    print("time: " + str(par[i].time) + ", loss: " + str(par[i].loss) + ", acc: " + str(np.exp(-par[i].loss)))
plt.clf()
plt.xlabel('time')
plt.ylabel('loss')
axes = plt.gca()
axes.set_xlim([x_bound[0],x_bound[1]])
axes.set_ylim([y_bound[0],y_bound[1]])
plt.plot(time[0:init_amount], loss[0:init_amount],'yo')
plt.plot(time[init_amount:], loss[init_amount:], 'o')
plt.plot(par_time, par_loss, 'ro')
plt.pause(float('inf'))
