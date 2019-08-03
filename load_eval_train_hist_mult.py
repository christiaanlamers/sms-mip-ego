import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

load_all = False

file_name = 'data/train_hist_best/stats.json'
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

#plt.plot(x,y_train,'red')
print("Online: " + str(max(y)))

orange, = plt.plot(x,y,'orange',label='State-of-the-art')

file_name = 'data/train_hist_best/skippy_tweaked_best_eval_train_hist.json'

eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))

y = eval_data[0][1]
print("Tweaked: " + str(max(y)))
blue, = plt.plot(x,y,'blue',label='Fine tuned search space')

if load_all:
    file_name = 'data/train_hist_best/skippy_test_train_hist_cut_eval_train_hist.json'

    eval_data = []
    with open(file_name) as f:
        for line in f:
            eval_data.append(json.loads(line))


    y = eval_data[0][1]
    print("Cut: " + str(max(y)))
    green, = plt.plot(x,y,'green',label='Pruning one node layers')

    file_name = 'data/train_hist_best/skippy_test_train_hist_aug_eval_train_hist.json'

    eval_data = []
    with open(file_name) as f:
        for line in f:
            eval_data.append(json.loads(line))


    y = eval_data[0][2]
    print("Data-aug: " + str(max(y)))
    purple, = plt.plot(x,y,'purple',label='Data augmentation')

    file_name = 'data/train_hist_best/skippy_test_train_hist_aug_tr_tw_eval_train_hist.json'

    eval_data = []
    with open(file_name) as f:
        for line in f:
            eval_data.append(json.loads(line))


    y = eval_data[0][2]
    print("Train-tweak: " + str(max(y)))
    red, = plt.plot(x,y,'red',label='Training schedule optimization')

    plt.legend(handles=[orange,blue,green,purple,red])
else:
    plt.legend(handles=[orange,blue])

plt.xlabel('epochs',size=20)
plt.ylabel('accuracy',size=20)

plt.tight_layout()
plt.show()
