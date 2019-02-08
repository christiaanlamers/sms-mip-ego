import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set(style="darkgrid")

import sys

try:
    zoom = str(sys.argv[2])
except:
    zoom = 0

display_time = True
display_max_data = True

file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

max_acc = 0
max_time = 0
max_acc_clamped = 0 #max in under 1 minutes
minute_ten_it = 0
all_max_data = []
for i in range(len(data)):
    clamp = 0
    for j in range(len(data[i][1])):
        if data[i][1][j] > 36000:
            break
        else:
            clamp = j
    if max(data[i][0][:clamp]) > max_acc_clamped:
        minute_ten_it = clamp
    max_acc_clamped = max(max_acc_clamped, max(data[i][0][:clamp]))
    for j in range(len(data[i][0])):
        if max_acc < data[i][0][j]:
            max_acc = data[i][0][j]
            max_time = data[i][1][j]
    max_data= []
    for j in range(len(data[i][0])):
        max_data.append(max(data[i][0][:j+1]))
    all_max_data.append(max_data)
    if display_time:
        if display_max_data:
            panda_data = pd.DataFrame(data={'time': data[i][1], 'validation accuracy':max_data})
            #plt.plot(data[i][1],max_data,label='par '+str(i))
        else:
            panda_data = pd.DataFrame(data={'time': data[i][1], 'validation accuracy':data[i][0]})
            #plt.plot(data[i][1],data[i][0],label='par '+str(i))
        #sns.regplot(x='time', y='validation accuracy', data=panda_data, color = 'r')
    else:
        if display_max_data:
            panda_data = pd.DataFrame(data={'iterations': [i for i in range(len(data[i][0]))], 'validation accuracy':max_data})
            #plt.plot(max_data,label='par '+str(i))
        else:
            panda_data = pd.DataFrame(data={'iterations': [i for i in range(len(data[i][0]))], 'validation accuracy':data[i][0]})
            #plt.plot(data[i][0],label='par '+str(i))
        #sns.regplot(x='iterations', y='validation accuracy', data=panda_data, color = 'r')
panda_indexes = []
panda_time = []
panda_max_acc = []
for i in range(len(data)):
    for j in range(len(data[i][0])):
        panda_indexes.append('par ' +str(i))
        panda_time.append(data[i][1][j])
        panda_max_acc.append(all_max_data[i][j])
all_panda_data =pd.DataFrame(data={'pareto front':panda_indexes,'time': panda_time, 'validation accuracy':panda_max_acc})
a = sns.relplot(x="time", y="validation accuracy", hue="pareto front",kind="line", data=all_panda_data)
if zoom:
    a.set(xlim=(0,20000),ylim=(0.9, 1.0))
else:
    a.set(xlim=(0,20000),ylim=(0.0, 1.0))
print("max accuracy:")
print(max_acc)
print("at time (seconds):")
print(max_time)
print("at time (minutes):")
print(max_time/60)
print("at time (hours):")
print(max_time/(60*60))
print("max accuracy in under 10 hours:")
print(max_acc_clamped)
print("10 hours was enough time for iteration number:")
print(minute_ten_it)
#plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.)
#if display_time:
#    plt.xlabel('time (s)')
#else:
#    plt.xlabel('# iterations')
#plt.ylabel('validation accuracy')
plt.show()
