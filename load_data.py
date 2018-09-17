import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import sys

from mipego.mipego import Solution
from mipego.Bi_Objective import *

class obj_func(object):
    def __init__(self, program):
        self.program = program
        
    def __call__(self, cfg, gpu_no):
        print("calling program with gpu "+str(gpu_no))
        cmd = ['python3', self.program, '--cfg', str(cfg), str(gpu_no)]
        outs = ""
        #outputval = 0
        outputval = ""
        try:
            outs = str(check_output(cmd,stderr=STDOUT, timeout=40000))
            if os.path.isfile(logfile): 
                with open(logfile,'a') as f_handle:
                    f_handle.write(outs)
            else:
                with open(logfile,'w') as f_handle:
                    f_handle.write(outs)
            outs = outs.split("\\n")
            
            #TODO_CHRIS hacky solution
            #outputval = 0
            #for i in range(len(outs)-1,1,-1):
            for i in range(len(outs)-1,-1,-1):
                #if re.match("^\d+?\.\d+?$", outs[-i]) is None:
                #CHRIS changed outs[-i] to outs[i]
                print(outs[i])
                if re.match("^\(\-?\d+\.?\d*\e?\+?\-?\d*\,\s\-?\d+\.?\d*\e?\+?\-?\d*\)$", outs[i]) is None:
                    #do nothing
                    a=1
                else:
                    #outputval = -1 * float(outs[-i])
                    outputval = outs[i]
            
            #if np.isnan(outputval):
            #    outputval = 0
        except subprocess.CalledProcessError as e:
            traceback.print_exc()
            print (e.output)
        except:
            print ("Unexpected error:")
            traceback.print_exc()
            print (outs)
            
            #outputval = 0
        #TODO_CHRIS hacky solution
        tuple_str1 = ''
        tuple_str2 = ''
        success = True
        i = 1
        try:
            while outputval[i] != ',':
                tuple_str1 += outputval[i]
                i += 1
            i += 1
            while outputval[i] != ')':
                tuple_str2 += outputval[i]
                i += 1
        except:
            print("error in receiving answer from gpu " + str(gpu_no))
            success = False
        try:
            tuple = (float(tuple_str1),float(tuple_str2),success)
        except:
            tuple = (0.0,0.0,False)
        #return outputval
        return tuple

if len(sys.argv) != 3 and len(sys.argv) != 5:
    print("usage: python3 load_data.py 'data_file_name.json' init_solution_number (optional: ref_time ref_loss)")
    exit(0)
file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

init_amount = int(sys.argv[2])

if len(sys.argv) == 5:
    ref_time = float(sys.argv[3])
    ref_loss = float(sys.argv[4])
else:
    ref_time = None
    ref_loss = None

conf_array = data[0]
fit_array = data[1]
time_array = data[2]
loss_array = data[3]
n_eval_array = data[4]
index_array = data[5]
name_array = data[6]

all_time_r2 = None
all_loss_r2 = None
if len(data) > 7:
    all_time_r2 = data[7]
    all_loss_r2 = data[8]


#print(data)
solutions = []
for i in range(len(conf_array)):
    solutions.append(Solution(x=conf_array[i],fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

print("len(solutions): " + str(len(solutions)))

pauser = 0.008

time = [x.time for x in solutions]
loss = [x.loss for x in solutions]

#print('time:')
#print(time)
#print('loss:')
#print(loss)
x_bound = min(0.0,min(time)),max(time)
y_bound = min(0.0,min(loss)),max(loss)

plt.ion()
for i in range(1,0):#len(solutions)):
    plt.clf()
    plt.xlabel('time')
    plt.ylabel('loss')
    axes = plt.gca()
    axes.set_xlim([x_bound[0],x_bound[1]])
    axes.set_ylim([y_bound[0],y_bound[1]])

    par = pareto(solutions[0:i])

    par_time = [x.time for x in par]
    par_loss = [x.loss for x in par]

    #matplotlib.rcParams['axes.unicode_minus'] = False
    #fig, ax = plt.subplots()
    plt.plot(time[0:min(i-1,init_amount)], loss[0:min(i-1,init_amount)],'yo')
    if i-1 > init_amount:
        plt.plot(time[init_amount:i-1], loss[init_amount:i-1], 'o')
    plt.plot(par_time, par_loss, 'ro')
    plt.plot(time[i], loss[i], 'go')
    #plt.set_title('sms-mip-ego')
    #plt.show()
    plt.pause(pauser)
par = pareto(solutions)
quicksort_par(par,0,len(par)-1)
par_time = [x.time for x in par]
par_loss = [x.loss for x in par]
HV = hyper_vol(par, solutions, ref_time, ref_loss)
objective = obj_func('./all-cnn_bi_mbarrier.py')
print("Hyper Volume:")
print(HV)
print("len pareto front:")
print(len(par))
print("paretofront:")
if all_time_r2 is not None and all_loss_r2 is not None:
    print("all_time_r2 average:")
    print(np.average(np.array(all_time_r2)))
    print("all_loss_r2 average:")
    print(np.average(np.array(all_loss_r2)))
for i in range(len(par)):
    print("time: " + str(par[i].time) + ", loss: " + str(par[i].loss) + ", acc: " + str(np.exp(-par[i].loss)))
#if all_time_r2 is not None and all_loss_r2 is not None:
#    print("all_time_r2:")
#    print(all_time_r2)
#    print("all_loss_r2:")
#    print(all_loss_r2)
#print(par)
#print(par[0].var_name)
#for i in range(1):#range(len(par)):
#    vec = par[i].to_dict()
#    print(vec)
#    print(objective(vec))
plt.clf()
#plt.xlabel('time')
#plt.ylabel('loss')
plt.xlabel('f_sphere_1')#CHRIS x^2
plt.ylabel('f_sphere_2')#(x-2)^2
axes = plt.gca()
axes.set_xlim([x_bound[0],x_bound[1]])
axes.set_ylim([y_bound[0],y_bound[1]])
plt.plot(time[0:init_amount], loss[0:init_amount],'yo')
plt.plot(time[init_amount:], loss[init_amount:], 'o')
plt.plot(par_time, par_loss, 'ro')
plt.pause(float('inf'))
