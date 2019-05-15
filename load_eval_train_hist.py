import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

file_name = str(sys.argv[1])
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))
plt.xlabel('epochs')
#plt.xlabel('time (hours)')
plt.ylabel('accuracy')

for i in eval_data:
    y = i[1]
    x = i[2]
    for j in range(len(x)):
        x[j] /= 60*60
    x = [i + 1 for i in range(len(y))]
    plt.plot(x,y)

plt.show()
