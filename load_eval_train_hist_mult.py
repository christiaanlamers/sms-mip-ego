import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

file_name = 'train_hist_best/stats.json'
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))


x= []
y=[]
y_train=[]
for i in eval_data[0]:
    y.append(1.0-i['validation_error'])
    y_train.append(1.0-i['train_error'])
    x.append(i['epoch_num'])
plt.xlabel('epochs')
plt.ylabel('accuracy')

#plt.plot(x,y_train,'red')
print("Online: " + str(max(y)))

plt.plot(x,y,'orange')

file_name = 'train_hist_best/skippy_tweaked_best_eval_train_hist.json'

eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')

y = eval_data[0][1]
print("Tweaked: " + str(max(y)))
plt.plot(x,y,'blue')

file_name = 'train_hist_best/skippy_test_train_hist_cut_eval_train_hist.json'

eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')


y = eval_data[0][1]
print("Cut: " + str(max(y)))
plt.plot(x,y,'green')

file_name = 'train_hist_best/skippy_test_train_hist_aug_eval_train_hist.json'

eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')


y = eval_data[0][2]
print("Data-aug: " + str(max(y)))
plt.plot(x,y,'purple')

file_name = 'train_hist_best/skippy_test_train_hist_aug_tr_tw_eval_train_hist.json'

eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')


y = eval_data[0][2]
print("Train-tweak: " + str(max(y)))
plt.plot(x,y,'red')

plt.show()
