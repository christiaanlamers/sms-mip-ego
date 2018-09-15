from scipy.stats import norm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

mu = 0.0
mu2= 1.0
sd = 1.0
sd2 = 1.0
alpha = float(sys.argv[1])

if alpha > 1.0 or alpha < 0:
    print('error: alpha for Lower Confidence bound must be between 0.0 and 1.0')
    exit()
elif alpha >= 0.5:
    exp,_ = norm.interval(alpha-(1.0-alpha),loc=mu,scale=sd)#CHRIS Lower Confidence Bound
    exp2,_ = norm.interval(alpha-(1.0-alpha),loc=mu2,scale=sd2)#CHRIS Lower Confidence Bound
else:
    _,exp = norm.interval(1.0-2.0*alpha,loc=mu,scale=sd)#CHRIS Lower Confidence Bound
    _,exp2 = norm.interval(1.0-2.0*alpha,loc=mu2,scale=sd2)#CHRIS Lower Confidence Bound

n = 1000
x = np.array(range(-3 * n,3 * n))/n
plt.plot(x, norm.pdf((x-mu)/sd)/sd)
plt.plot(x, norm.pdf((x-mu2)/sd2)/sd2,'g')
plt.plot(exp,norm.pdf((exp-mu)/sd)/sd,'ro')
plt.plot(exp2,norm.pdf((exp2-mu2)/sd2)/sd2,'yo')
plt.show()
