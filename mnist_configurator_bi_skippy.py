#!/usr/bin/env python3.0
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:38:25 2017

@author: wangronin & Bas van Stein
"""

import pdb

import subprocess, os, sys
from subprocess import STDOUT, check_output
import numpy as np
import time

import gputil as gp
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace
import re
import traceback
import time

np.random.seed(42)

#--------------------------- Configuration settings --------------------------------------
# TODO: implement parallel execution of model
n_step = 110
n_init_sample = 90
verbose = True
save = False
logfile = 'mnist.log'
class obj_func(object):
    def __init__(self, program):
        self.program = program
        
    def __call__(self, cfg, gpu_no):
        print("calling program with gpu "+str(gpu_no))
        cmd = ['python3', self.program, '--cfg', str(cfg), str(gpu_no)]
        #outputval = 0
        outputval = ""
        outs = ""
        try:
            outs = str(check_output(cmd,stderr=None, timeout=40000))#CHRIS stderr=None was stderr=STDOUT we don't want warnings because they mess up the output
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
                #if re.match("[\s\S]*ResourceExhaustedError[\s\S]*", outs[i]) is not None:
                    #print('GPU resource exhausted, penalty returned')
                    #return 1000000000.0, 5.0, True
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
            success = True #CHRIS simply give large penalty in case of failure instead of setting success to False
            tuple_str1 = '1000000000'
            tuple_str2 = '5'
        tuple = (float(tuple_str1),float(tuple_str2),success)
        #return outputval
        return tuple


#define the search space.
#objective = obj_func('./all-cnn_bi.py')
activation_fun = ["softmax"]
activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

filters = OrdinalSpace([10, 600], 'filters') * 14
kernel_size = OrdinalSpace([1, 7], 'k') * 14
strides = OrdinalSpace([1, 5], 's') * 7
stack_sizes = OrdinalSpace([0, 6], 'stack') * 7

activation = NominalSpace(activation_fun_conv, "activation")  # activation function
activation_dense = NominalSpace(activation_fun, "activ_dense") # activation function for dense layer
step = NominalSpace([True, False], "step")  # step
global_pooling = NominalSpace([True, False], "global_pooling")  # global_pooling

#skippy parameters
skints = OrdinalSpace([0, 2**50-1], 'skint') * 3#CHRIS TODO tweak this
skst = OrdinalSpace([2, 10], 'skst') * 3
dense_size = OrdinalSpace([0, 1200], 'dense_size')*2
no_pooling = NominalSpace([True, False], "no_pooling")
#skippy parameters

drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 8        # drop_out rate
lr_rate = ContinuousSpace([1e-4, 1.0e-0], 'lr')        # learning rate
l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer


search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skints * skst * dense_size * no_pooling


print('starting program...')    
#available_gpus = gp.getAvailable(limit=2)
available_gpus = gp.getAvailable(limit=5)

ignore_gpu = []
if len(sys.argv) > 1:
    for i in range(1,int(len(sys.argv))):
        print(int(sys.argv[i]))
        ignore_gpu.append(int(sys.argv[i]))
        try:
            available_gpus.remove(int(sys.argv[i]))
        except:
            pass
#try:
#available_gpus.remove(0)#CHRIS gpu 0 and 5 are differen gpu types on duranium since they are faster, timing will be unreliable, so remove them from list
#except:
#pass
#try:
#available_gpus.remove(5)
#except:
#pass
print(available_gpus)

n_job = max(min(5,len(available_gpus)),1)


# use random forest as the surrogate model
#CHRIS two surrogate models are needed
time_model = RandomForest(levels=search_space.levels,n_estimators=100)
loss_model = RandomForest(levels=search_space.levels,n_estimators=100)
opt = mipego(search_space, objective, time_model, loss_model, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, max_iter=n_step, 
                 infill='HVI', n_init_sample=n_init_sample, n_point=1, n_job=n_job,
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES', 
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=available_gpus, bi=True, save_name='data_skippy_cifar10_large_sp',ref_time=None,ref_loss=None,ignore_gpu=ignore_gpu)

#ref_time=3000.0,ref_loss=3.0

incumbent, stop_dict = opt.run()
#print('incumbent #TODO_CHRIS makes no sense for now:')
#for x in incumbent:
#    try:
#        print(str(x) + ':' + str(incumbent[x]))
#    except:
#        continue
#print ('stop_dict:')
#for x in stop_dict:
#    try:
#        print(str(x) + ':' + str(stop_dict[x]))
#    except:
#        continue

