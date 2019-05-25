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
import math

import json
from mipego.mipego import Solution

np.random.seed(42)

#--------------------------- Configuration settings --------------------------------------
# TODO: implement parallel execution of model
n_step = 780
n_init_sample = 20
eval_epochs = 10
verbose = True
save = False
logfile = 'mnist.log'
class obj_func(object):
    def __init__(self, program, save_name='test'):
        self.program = program
        self.save_name = save_name
        
    def __call__(self, cfg, gpu_no,eval_epochs,save_name,data_augmentation,use_validation):
        with open(self.save_name + '_thread_log.json', 'a') as outfile:
            outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 1\n')
        print("calling program with gpu "+str(gpu_no))
        cmd = ['python3', self.program, '--cfg', str(cfg), str(gpu_no),str(eval_epochs),str(save_name),str(data_augmentation),str(use_validation)]
        #outputval = 0
        outputval = ""
        outs = ""
        with open(self.save_name + '_thread_log.json', 'a') as outfile:
            outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 2\n')
        try:
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 3\n')
            outs = str(check_output(cmd,stderr=STDOUT, timeout=40000))#CHRIS stderr=None was stderr=STDOUT we don't want warnings because they mess up the output
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 4\n')
            if os.path.isfile(logfile): 
                with open(logfile,'a') as f_handle:
                    f_handle.write(outs)
            else:
                with open(logfile,'w') as f_handle:
                    f_handle.write(outs)
            outs = outs.split("\\n")
            #print('this is outs:')
            #print(outs)
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 5\n')
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
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 6\n')
            #if np.isnan(outputval):
            #    outputval = 0
        except subprocess.CalledProcessError as e:
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 6a error\n')
            traceback.print_exc()
            print (e.output)
        except:
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 6b error\n')
            print ("Unexpected error:")
            traceback.print_exc()
            print (outs)
            
            #outputval = 0

        with open(self.save_name + '_thread_log.json', 'a') as outfile:
            outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 7\n')
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
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 8a\n')
        except:
            with open(self.save_name + '_thread_log.json', 'a') as outfile:
                outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 8b error\n')
            print("error in receiving answer from gpu " + str(gpu_no))
            success = True #CHRIS simply give large penalty in case of failure instead of setting success to False
            tuple_str1 = '80000'#CHRIS 2 times timeout value
            tuple_str2 = str(-1 * math.log(0.05))#CHRIS half the accuracy of random guessing
        tuple = (float(tuple_str1),float(tuple_str2),success)
        #return outputval
        with open(self.save_name + '_thread_log.json', 'a') as outfile:
            outfile.write('thread ' + str(gpu_no) + ': step 3 gpu 3 obj_func 9\n')
        return tuple


#define the search space.
save_name = '../../../data/s0315435/data_skippy_cifar10_better_data_augmentation_big_one'
objective = obj_func('./all_cnn_bi_skippy_aug.py',save_name=save_name)
activation_fun = ["softmax"]
activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

filters = OrdinalSpace([10, 600], 'filters') * 14
kernel_size = OrdinalSpace([1, 16], 'k') * 14#CHRIS tweaked
strides = OrdinalSpace([1, 4], 's') * 7#CHRIS tweaked TODO maybe limit to max of 3, because now the image is reduces too soon (used to be max 10) CHRIS tweaked again CHRIS tweaked a third time
stack_sizes = OrdinalSpace([0, 7], 'stack') * 7#[0,2] should be [0,7]

activation = NominalSpace(activation_fun_conv, "activation")  # activation function
activation_dense = NominalSpace(activation_fun, "activ_dense") # activation function for dense layer
step = NominalSpace([True, False], "step")  # step
global_pooling = NominalSpace([True,False], "global_pooling")  # global_pooling#CHRIS TODO removed False

#skippy parameters
skstart = OrdinalSpace([0, 7], 'skstart') * 5
skstep = OrdinalSpace([1, 10], 'skstep') * 5#CHRIS a skip step of 1 means no skip connection#OrdinalSpace([1, 10], 'skst') * 3
max_pooling = NominalSpace([True, False], "max_pooling")
dense_size = OrdinalSpace([0,4000],'dense_size')*2#CHRIS tweaked
#skippy parameters

drop_out = ContinuousSpace([0.0, .42], 'dropout') * 10        # drop_out rate #tweaked again, min used to be 1e-5 max used to be 0.9
lr_rate = ContinuousSpace([1e-4, 1.0e-2], 'lr')        # learning rate#CHRIS tweaked #CHRIS tweaked again for data augmentation (max used to be 1.0e-2)
l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer

batch_size_sp = OrdinalSpace([50, 200], 'batch_size_sp')#CHRIS tweaked again: added to search space

#augmented parameters
featurewise_center = NominalSpace([True,False], "featurewise_center")
samplewise_center = NominalSpace([True,False], "samplewise_center")
featurewise_std_normalization = NominalSpace([True,False], "featurewise_std_normalization")
samplewise_std_normalization = NominalSpace([True,False], "samplewise_std_normalization")
zca_epsilon = ContinuousSpace([0.5e-6, 2e-6], 'zca_epsilon')
zca_whitening = NominalSpace([True,False], "zca_whitening")
rotation_range = OrdinalSpace([0, 360], 'rotation_range')
width_shift_range = ContinuousSpace([0.0, 1.0], 'width_shift_range')
height_shift_range = ContinuousSpace([0.0, 1.0], 'height_shift_range')
shear_range = ContinuousSpace([0.0, 45.0], 'shear_range')
zoom_range = ContinuousSpace([0.0, 1.0], 'zoom_range')
channel_shift_range = ContinuousSpace([0.0, 1.0], 'channel_shift_range')
fill_mode = NominalSpace(["constant","nearest","reflect","wrap"], "fill_mode")
cval = ContinuousSpace([0.0, 1.0], 'cval')
horizontal_flip = NominalSpace([True,False], "horizontal_flip")
vertical_flip = NominalSpace([True,False], "vertical_flip")
#augmented parameters


search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skstart * skstep * max_pooling * dense_size * batch_size_sp * featurewise_center * samplewise_center * featurewise_std_normalization * samplewise_std_normalization * zca_epsilon * zca_whitening * rotation_range * width_shift_range * height_shift_range * shear_range * zoom_range * channel_shift_range * fill_mode * cval * horizontal_flip * vertical_flip

print('starting program...')    
#available_gpus = gp.getAvailable(limit=2)
gpu_limit = 16
available_gpus = gp.getAvailable(limit=gpu_limit)

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

n_job = max(min(gpu_limit,len(available_gpus)),1)


# use random forest as the surrogate model
#CHRIS two surrogate models are needed
time_model = RandomForest(levels=search_space.levels,n_estimators=10)
loss_model = RandomForest(levels=search_space.levels,n_estimators=10)
opt = mipego(search_space, objective, time_model, loss_model, ftarget=None,
                 minimize=True, noisy=False, max_eval=None, max_iter=n_step, 
                 infill='HVI', n_init_sample=n_init_sample, n_point=1, n_job=n_job,
                 n_restart=None, max_infill_eval=None, wait_iter=3, optimizer='MIES',
                 log_file=None, data_file=None, verbose=False, random_seed=None,
                 available_gpus=available_gpus, bi=True, save_name=save_name,ref_time=None,ref_loss=None,ignore_gpu=ignore_gpu,eval_epochs=eval_epochs,data_augmentation=True,use_validation = True)

#ref_time=3000.0,ref_loss=3.0

#incumbent, stop_dict = opt.run() #CHRIS opt.run() does not return anything anymore
#CHRIS restart code
if False:
    with open('data_skippy_cifar10_big_one_data_augmentation_intermediate_restarted1.json') as f:
        for line in f:
            data = json.loads(line)

    conf_array = data[0]
    fit_array = data[1]
    time_array = data[2]
    loss_array = data[3]
    n_eval_array = data[4]
    index_array = data[5]
    name_array = data[6]

    all_time_r2 = data[7]
    all_loss_r2 = data[8]

    surr_time_fit_hist = data[9]
    surr_time_mies_hist = data[10]
    surr_loss_fit_hist = data[11]
    surr_loss_mies_hist = data[12]
    time_between_gpu_hist = data[13]


    solutions = []
    for i in range(len(conf_array)):
        conf_x = [conf_array[i][j] for j in name_array[i]]
        solutions.append(Solution(x=conf_x,fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

    opt.data=solutions

    for i in range(len(opt.data)):
        opt.data[i].fitness = fit_array[i]
        opt.data[i].time = time_array[i]
        opt.data[i].loss = loss_array[i]
        opt.data[i].n_eval = n_eval_array[i]
        opt.data[i].index = index_array[i]
        opt.data[i].var_name = name_array[i]

    opt.all_time_r2 = all_time_r2
    opt.all_loss_r2 = all_loss_r2
    opt.surr_time_fit_hist = surr_time_fit_hist
    opt.surr_time_mies_hist = surr_time_mies_hist
    opt.surr_loss_fit_hist = surr_loss_fit_hist
    opt.surr_loss_mies_hist = surr_loss_mies_hist
    opt.time_between_gpu_hist = time_between_gpu_hist

    opt.n_left = opt.max_iter - len(opt.data)+opt.n_init_sample
    opt.iter_count = len(opt.data)-opt.n_init_sample
    opt.eval_count = len(opt.data)-opt.n_init_sample

    opt.run(restart=True)
else:
    opt.run()
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

