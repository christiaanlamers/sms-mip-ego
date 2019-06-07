import matplotlib.pyplot as plt
import math

def schedule (initial_lrate,drop,epochs_drop,epochs):
    #lrate = initial_lrate * math.pow(exp_decay,epochs)*lin_decay*epochs
    lrate = initial_lrate * math.pow(drop, math.floor((1+epochs)/epochs_drop))
    return lrate

x = [i for i in range(100)]
y = [0 for i in range(100)]
initial_lrate = 1.0
drop = 0.9
epochs_drop = 10.0
for i in range(len(x)):
    y[i] = schedule(initial_lrate,drop,epochs_drop,x[i])

plt.plot(x,y)
plt.show()
