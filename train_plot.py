import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

for i in range(len(data)):
    plt.plot(data[i][1],data[i][0],label='par '+str(i))
plt.legend(bbox_to_anchor=(1.03, 1), loc=2, borderaxespad=0.)
plt.show()
