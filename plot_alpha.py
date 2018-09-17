import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
#y = np.array([6674.096851731184,6346.532457807667,7107.038988600756,6452.650101629087,7009.896575942308,7188.037248363666 ,6614.584177497748,7563.809106502014,7021.410802111])
y = np.array([23, 33, 19, 14, 19, 9, 10, 10, 11])
fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit)

plt.clf()
#plt.xlabel('time')
#plt.ylabel('loss')
plt.xlabel('alpha')#CHRIS x^2
plt.ylabel('# pareto front')#(x-2)^2
axes = plt.gca()
#axes.set_xlim([x_bound[0],x_bound[1]])
#axes.set_ylim([y_bound[0],y_bound[1]])
#plt.plot(x, y, 'o')
plt.plot(x,y, 'o', x, fit_fn(x), '--k')
plt.show()
