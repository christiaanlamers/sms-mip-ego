import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

if len(sys.argv) != 3:
    print("usage: python3 load_eval_train_hist <file_name> <train_acc_in_data> (0,1)")
    exit(0)
file_name = str(sys.argv[1])
train_accuracy = int(sys.argv[2])
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')

if train_accuracy:
    if len(eval_data[0]) != 4:
        print("error, no train accuracy data in file")
        exit(0)
    for i in eval_data:
        y_train = i[1]
        y_val = i[2]
        x = i[3]
        for j in range(len(x)):
            x[j] /= 60*60
        x = [i + 1 for i in range(len(y_train))]
        plt.plot(x,y_train,'r')
        plt.plot(x,y_val,'g')
else:
    if len(eval_data[0]) != 3:
        print("error, train accuracy present in data file")
        exit(0)
    for i in eval_data:
        y = i[1]
        x = i[2]
        for j in range(len(x)):
            x[j] /= 60*60
        x = [i + 1 for i in range(len(y))]
        plt.plot(x,y)

plt.show()
