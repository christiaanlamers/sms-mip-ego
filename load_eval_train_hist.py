import json
import sys
import matplotlib
import matplotlib.pyplot as plt

file_name = str(sys.argv[1])
eval_data = []
with open(file_name) as f:
    for line in f:
        eval_data.append(json.loads(line))

for i in eval_data:
    plt.plot(i[2],i[1])

plt.show()
