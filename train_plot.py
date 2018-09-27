import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

display_time = False

file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

max_acc = 0
max_time = 0
max_acc_clamped = 0 #max in under 10 minutes
minute_ten_it = 0
for i in range(len(data)):
    clamp = 0
    for j in range(len(data[i][1])):
        if data[i][1][j] > 600:
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
    if display_time:
        plt.plot(data[i][1],data[i][0],label='par '+str(i))
    else:
        plt.plot(data[i][0],label='par '+str(i))
print("max accuracy:")
print(max_acc)
print("at time (seconds):")
print(max_time)
print("at time (minutes):")
print(max_time/60)
print("at time (hours):")
print(max_time/(60*60))
print("max accuracy in under 10 minutes:")
print(max_acc_clamped)
print("10 minutes was enough time for iteration number:")
print(minute_ten_it)
plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.)
if display_time:
    plt.xlabel('time (s)')
else:
    plt.xlabel('# iterations')
plt.ylabel('validation accuracy')
plt.show()
