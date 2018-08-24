from __future__ import print_function

import numpy as np
np.random.seed(43)
import tensorflow as tf
tf.set_random_seed(43)

import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import os
import sys
import pandas as pd
import keras.backend as K
import math
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

import time #CHRIS added to measure runtime of training
import deap.benchmarks
import numbers

from subprocess import call #CHRIS added to use shell commands


def CNN_conf(cfg):
    entry_no = len(cfg)
    #for x in cfg:
    #    if not(isinstance(cfg[x], numbers.Number) and cfg[x] is not None):#TODO_CHRIS maybe remove this if bounds are set ok
    #        entry_no -= 1
    with open("moptfilt_Temp_result.txt", "w") as text_file:
        print("{}".format(float(entry_no)), file=text_file)
        for x in cfg:
            #if isinstance(cfg[x], numbers.Number) and cfg[x] is not None:#TODO_CHRIS maybe remove this if bounds are set ok
            print("{}".format(cfg[x]), file=text_file)
    #tuple = deap.benchmarks.schaffer_mo(vector)
    call(["./MI_test_functions/moptfilt/moptfilt","moptfilt_Temp_result.txt","moptfilt_Temp_result_out.txt"])
    lines = [line.rstrip('\n') for line in open("moptfilt_Temp_result_out.txt")]
    print("lines:")
    print(lines)
    lines[0] = float(lines[0])
    lines[1] = float(lines[1])
    #if lines[0] == float('inf'):
    #    lines[0] = sys.float_info.max
    #if lines[0] == -float('inf'):
    #    lines[0] = -sys.float_info.max
    #if lines[1] == float('inf'):
    #    lines[1] = sys.float_info.max
    #if lines[1] == -float('inf'):
    #    lines[1] = -sys.float_info.max
    tuple = (lines[0],lines[1])
    return tuple



#system arguments (configuration)
if len(sys.argv) > 2 and sys.argv[1] == '--cfg':
    cfg = eval(sys.argv[2])
    if len(sys.argv) > 3:
        gpu = sys.argv[3]
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    print(CNN_conf(cfg))
    K.clear_session()
