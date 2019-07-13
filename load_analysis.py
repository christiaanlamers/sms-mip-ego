import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from all_cnn_bi_skippy import inv_gray, Skip_manager,CNN_conf
from keras.utils import plot_model
import sys
from mipego.mipego import Solution
from mipego.Bi_Objective import *
import math
from scipy.interpolate import UnivariateSpline
import scipy.special
import numbers
from pandas.plotting import parallel_coordinates
import sklearn
from sklearn.cluster import KMeans, DBSCAN
import sklearn.metrics as sm
from apyori import apriori
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

cut = '_cut_' in  str(sys.argv[1])#needed for derived data such as depth, in case of the cut method, the network is less deep
train_tweak = 'train_tweak' in str(sys.argv[1])
if cut:
    print("cut!")
    disfunc_time = 200000#80000 #200000 #CHRIS penalty value given to a disfuncitonal network. This differs per experiment
elif train_tweak:
    print("train tweak!")
    disfunc_time = 800000
else:
    print("no cut!")
    disfunc_time = 80000 #200000 #CHRIS penalty value given to a disfuncitonal network. This differs per experiment
max_stack = 7 #the number of stacks in the construction method
CIFAR10 = True #do we use CIFAR-10 or MNIST?
img_dim = 32 #CIFAR-10 has 32x32 images

do_spline_fit = False
do_parallel_plot = True
do_pairgrid = False
do_correlations = False
do_k_means = False
do_dbscan = False
do_rule_finding = False
do_feature_imp = False
do_sens_analysis = False

file_name = str(sys.argv[1])
with open(file_name) as f:
    for line in f:
        data = json.loads(line)

conf_array = data[0]
fit_array = data[1]
time_array = data[2]
loss_array = data[3]
n_eval_array = data[4]
index_array = data[5]
name_array = data[6]

all_time_r2 = []
all_loss_r2 = []
if len(data) > 7:
    all_time_r2 = data[7]
    all_loss_r2 = data[8]

surr_time_fit_hist = []
surr_time_mies_hist = []
surr_loss_fit_hist = []
surr_loss_mies_hist = []
time_between_gpu_hist = []
if len(data) > 9:
    surr_time_fit_hist = data[9]
    surr_time_mies_hist = data[10]
    surr_loss_fit_hist = data[11]
    surr_loss_mies_hist = data[12]
if len(data) > 13:
    time_between_gpu_hist = data[13]


#print(data)
solutions = []
for i in range(len(conf_array)):
    conf_x = [conf_array[i][j] for j in name_array[i]]
    if time_array[i] < disfunc_time:#TODO CHRIS filter out disfunciontal network, note that the time penalty value differs per research instance
        solutions.append(Solution(x=conf_x,fitness=fit_array[i],n_eval=n_eval_array[i],index=index_array[i],var_name=name_array[i],loss=loss_array[i],time=time_array[i]))

#TODO test code, remove this
#vla = {'stack_1': 1, 'k_11': 13, 'dropout_3': 0.8165875526780366, 'max_pooling': True, 'skstep_1': 6, 'dropout_1': 0.23195911906645486, 'dropout_0': 0.8979110939293858, 's_6': 1, 'dropout_4': 0.4075181725682306, 'stack_3': 1, 'global_pooling': True, 'stack_0': 0, 'skstart_2': 1, 'filters_6': 43, 'k_9': 8, 'k_8': 2, 'dropout_6': 0.7253793783496744, 'k_12': 3, 's_0': 6, 'k_3': 8, 'filters_2': 45, 'filters_0': 287, 'skstart_1': 3, 'step': False, 'filters_3': 477, 'k_4': 10, 'filters_4': 15, 'filters_10': 230, 'dropout_9': 0.8933940350619436, 'filters_7': 131, 'filters_1': 389, 'skstart_4': 0, 'dropout_7': 0.06222967681587999, 'k_5': 11, 'skstep_0': 7, 'skstep_3': 5, 'activation': 'elu', 'dense_size_0': 2414, 'filters_8': 382, 'dropout_5': 0.021260790909403388, 'dense_size_1': 3980, 's_1': 4, 'stack_4': 6, 'filters_12': 263, 'stack_6': 6, 'skstart_0': 7, 'stack_2': 0, 'stack_5': 0, 's_2': 8, 'dropout_2': 0.4709580590734798, 'filters_11': 354, 'k_7': 11, 'filters_13': 84, 'l2': 0.00025394282998525213, 'k_2': 4, 'k_0': 15, 'skstart_3': 3, 'filters_5': 144, 's_4': 2, 's_3': 4, 'k_1': 14, 'k_10': 2, 'skstep_4': 3, 'activ_dense': 'softmax', 'filters_9': 203, 'k_6': 10, 'lr': 0.0019219792048921292, 'skstep_2': 3, 'k_13': 2, 's_5': 1, 'dropout_8': 0.8643908357346506}
#solutions = [Solution(x=[vla[j] for j in name_array[0]],fitness=1,n_eval=1,index=index_array[0],var_name=name_array[0],loss=1,time=1)]
#TODO end test code removal

print("len(solutions): " + str(len(solutions)))
#print([i.to_dict() for i in solutions])
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
acc_pivot = 0.7#0.4#0.86
data_lib = {}
data_lib_good = {}
data_lib_bad = {}
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
data_lib['acc'] = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
data_lib['time'] = [i.time for i in solutions]
data_lib_good['acc'] = [np.exp(-i.loss) for i in solutions if np.exp(-i.loss) >= acc_pivot]#[i.time for i in solutions]
data_lib_good['time'] = [i.time for i in solutions if np.exp(-i.loss) >= acc_pivot]
data_lib_bad['acc'] = [np.exp(-i.loss) for i in solutions if np.exp(-i.loss) < acc_pivot]#[i.time for i in solutions]
data_lib_bad['time'] = [i.time for i in solutions if np.exp(-i.loss) < acc_pivot]
#n_points = 1000
stopper = 0

print("number of solutions: " + str(len([i.time for i in solutions])))
print("number of good solutions: " + str(len([i.time for i in solutions if np.exp(-i.loss) >= acc_pivot])))
print("number of bad solutions: " + str(len([i.time for i in solutions if np.exp(-i.loss) < acc_pivot])))

for i in name_array[0]:
    print(i)
    x = []
    x_good = []
    x_bad = []
    #y = []
    for j in solutions:
        #if j.to_dict()['lr'] < 0.01:
        x.append(j.to_dict()[i])
        if np.exp(-j.loss) >= acc_pivot:
            x_good.append(j.to_dict()[i])
        else:
            x_bad.append(j.to_dict()[i])
        #y.append(np.exp(-j.loss))
    if isinstance(x[0], numbers.Number): #TODO CHRIS remove this and catch non number related errors
        data_lib[i] = x
        data_lib_good[i] = x_good
        data_lib_bad[i] = x_bad
    elif x[0] == "elu" or x[0] == "relu" or x[0] == "tanh" or x[0] == "sigmoid" or x[0] == "selu":
        elu = []
        relu = []
        tanh= []
        sigmoid = []
        selu = []
        activation = []
        for j in x:
            elu.append(j == "elu")
            relu.append(j=="relu")
            tanh.append(j=="tanh")
            sigmoid.append(j=="sigmoid")
            selu.append(j=="selu")
            activation.append(j)
        data_lib["elu"] = [int(i) for i in elu]
        data_lib["relu"] = [int(i) for i in relu]
        data_lib["tanh"] = [int(i) for i in tanh]
        data_lib["sigmoid"] = [int(i) for i in sigmoid]
        data_lib["selu"] = [int(i) for i in selu]
        data_lib["activation"] = activation

        elu = []
        relu = []
        tanh= []
        sigmoid = []
        selu = []
        activation = []
        for j in x_good:
            elu.append(j == "elu")
            relu.append(j=="relu")
            tanh.append(j=="tanh")
            sigmoid.append(j=="sigmoid")
            selu.append(j=="selu")
            activation.append(j)
        data_lib_good["elu"] = [int(i) for i in elu]
        data_lib_good["relu"] = [int(i) for i in relu]
        data_lib_good["tanh"] = [int(i) for i in tanh]
        data_lib_good["sigmoid"] = [int(i) for i in sigmoid]
        data_lib_good["selu"] = [int(i) for i in selu]
        data_lib_good["activation"] = activation
        
        elu = []
        relu = []
        tanh= []
        sigmoid = []
        selu = []
        activation = []
        for j in x_bad:
            elu.append(j == "elu")
            relu.append(j=="relu")
            tanh.append(j=="tanh")
            sigmoid.append(j=="sigmoid")
            selu.append(j=="selu")
            activation.append(j)
        data_lib_bad["elu"] = [int(i) for i in elu]
        data_lib_bad["relu"] = [int(i) for i in relu]
        data_lib_bad["tanh"] = [int(i) for i in tanh]
        data_lib_bad["sigmoid"] = [int(i) for i in sigmoid]
        data_lib_bad["selu"] = [int(i) for i in selu]
        data_lib_bad["activation"] = activation
    elif x[0] == "softmax":
        pass
        #softmax = []
        #for j in x:
        #    softmax.append(j=="softmax")
        #data_lib["softmax"] = softmax
        #
        #softmax = []
        #for j in x_good:
        #    softmax.append(j=="softmax")
        #data_lib_good["softmax"] = softmax
        #
        #softmax = []
        #for j in x_bad:
        #    softmax.append(j=="softmax")
        #data_lib_bad["softmax"] = softmax
    elif x[0] == "constant" or x[0] == "nearest" or x[0] == "reflect" or x[0] == "wrap":
        constant = []
        nearest = []
        reflect= []
        wrap = []
        fill_mode = []
        for j in x:
            constant.append(j == "constant")
            nearest.append(j=="nearest")
            reflect.append(j=="reflect")
            wrap.append(j=="wrap")
            fill_mode.append(j)
        data_lib["constant"] = [int(i) for i in constant]
        data_lib["nearest"] = [int(i) for i in nearest]
        data_lib["reflect"] = [int(i) for i in reflect]
        data_lib["wrap"] = [int(i) for i in wrap]
        data_lib["fill_mode"] = fill_mode

        constant = []
        nearest = []
        reflect= []
        wrap = []
        fill_mode = []
        for j in x_good:
            constant.append(j == "constant")
            nearest.append(j=="nearest")
            reflect.append(j=="reflect")
            wrap.append(j=="wrap")
            fill_mode.append(j)
        data_lib_good["constant"] = [int(i) for i in constant]
        data_lib_good["nearest"] = [int(i) for i in nearest]
        data_lib_good["reflect"] = [int(i) for i in reflect]
        data_lib_good["wrap"] = [int(i) for i in wrap]
        data_lib_good["fill_mode"] = fill_mode
        
        constant = []
        nearest = []
        reflect= []
        wrap = []
        fill_mode = []
        for j in x_bad:
            constant.append(j == "constant")
            nearest.append(j=="nearest")
            reflect.append(j=="reflect")
            wrap.append(j=="wrap")
            fill_mode.append(j)
        data_lib_bad["constant"] = [int(i) for i in constant]
        data_lib_bad["nearest"] = [int(i) for i in nearest]
        data_lib_bad["reflect"] = [int(i) for i in reflect]
        data_lib_bad["wrap"] = [int(i) for i in wrap]
        data_lib_bad["fill_mode"] = fill_mode
    elif x[0] == "SGD" or x[0] == "RMSprop" or x[0] == "Adagrad" or x[0] == "Adadelta" or x[0] == "Adam" or x[0] == "Adamax" or x[0] == "Nadam":
        SGD = []
        RMSprop = []
        Adagrad = []
        Adadelta = []
        Adam = []
        Adamax = []
        Nadam = []
        optimizer = []
        for j in x:
            SGD.append(j == "SGD")
            RMSprop.append(j=="RMSprop")
            Adagrad.append(j=="Adagrad")
            Adadelta.append(j=="Adadelta")
            Adam.append(j=="Adam")
            Adamax.append(j=="Adamax")
            Nadam.append(j=="Nadam")
            optimizer.append(j)
        data_lib["SGD"] = [int(i) for i in SGD]
        data_lib["RMSprop"] = [int(i) for i in RMSprop]
        data_lib["Adagrad"] = [int(i) for i in Adagrad]
        data_lib["Adadelta"] = [int(i) for i in Adadelta]
        data_lib["Adam"] = [int(i) for i in Adam]
        data_lib["Adamax"] = [int(i) for i in Adamax]
        data_lib["Nadam"] = [int(i) for i in Nadam]
        data_lib["optimizer"] = optimizer
        
        SGD = []
        RMSprop = []
        Adagrad = []
        Adadelta = []
        Adam = []
        Adamax = []
        Nadam = []
        optimizer = []
        for j in x_good:
            SGD.append(j == "SGD")
            RMSprop.append(j=="RMSprop")
            Adagrad.append(j=="Adagrad")
            Adadelta.append(j=="Adadelta")
            Adam.append(j=="Adam")
            Adamax.append(j=="Adamax")
            Nadam.append(j=="Nadam")
            optimizer.append(j)
        data_lib_good["SGD"] = [int(i) for i in SGD]
        data_lib_good["RMSprop"] = [int(i) for i in RMSprop]
        data_lib_good["Adagrad"] = [int(i) for i in Adagrad]
        data_lib_good["Adadelta"] = [int(i) for i in Adadelta]
        data_lib_good["Adam"] = [int(i) for i in Adam]
        data_lib_good["Adamax"] = [int(i) for i in Adamax]
        data_lib_good["Nadam"] = [int(i) for i in Nadam]
        data_lib_good["optimizer"] = optimizer
        
        SGD = []
        RMSprop = []
        Adagrad = []
        Adadelta = []
        Adam = []
        Adamax = []
        Nadam = []
        optimizer = []
        for j in x_bad:
            SGD.append(j == "SGD")
            RMSprop.append(j=="RMSprop")
            Adagrad.append(j=="Adagrad")
            Adadelta.append(j=="Adadelta")
            Adam.append(j=="Adam")
            Adamax.append(j=="Adamax")
            Nadam.append(j=="Nadam")
            optimizer.append(j)
        data_lib_bad["SGD"] = [int(i) for i in SGD]
        data_lib_bad["RMSprop"] = [int(i) for i in RMSprop]
        data_lib_bad["Adagrad"] = [int(i) for i in Adagrad]
        data_lib_bad["Adadelta"] = [int(i) for i in Adadelta]
        data_lib_bad["Adam"] = [int(i) for i in Adam]
        data_lib_bad["Adamax"] = [int(i) for i in Adamax]
        data_lib_bad["Nadam"] = [int(i) for i in Nadam]
        data_lib_bad["optimizer"] = optimizer
    else:
        print("error, unknown feature!")
    #print(x)
    if do_spline_fit:
        try:
            plt.xlabel(i)
            plt.ylabel('accuracy')
            #plt.ylabel('time')
            plt.plot(x,y, 'o')
        except:
            print('could not plot data')
        try:
            spl = UnivariateSpline(x, y)
            max_x = max(x)
            x_model = [k*max_x/n_points for k in range(n_points)]
            y_model = [spl(k) for k in x_model]
            plt.plot(x_model,y_model,color="red")
        except:
            print('could not fit a spline')
        try:
            plt.show()
        except:
            pass

#add extra features to data_lib TODO test this code
if not train_tweak:
    depth = np.array([0] * len(data_lib["stack_0"]))
    num_features = np.array([0] * len(data_lib["stack_0"]))
    img_size = np.array([img_dim] * len(data_lib["stack_0"]))
    avg_dropout = np.array([j for j in data_lib["dropout_0"]])
    dropout_norm = np.array([1] * len(data_lib["stack_0"]))

    depth_good = np.array([0] * len(data_lib_good["stack_0"]))
    num_features_good = np.array([0] * len(data_lib_good["stack_0"]))
    img_size_good = np.array([img_dim] * len(data_lib_good["stack_0"]))
    avg_dropout_good = np.array([j for j in data_lib_good["dropout_0"]])
    dropout_norm_good = np.array([1] * len(data_lib_good["stack_0"]))

    depth_bad = np.array([0] * len(data_lib_bad["stack_0"]))
    num_features_bad = np.array([0] * len(data_lib_bad["stack_0"]))
    img_size_bad = np.array([img_dim] * len(data_lib_bad["stack_0"]))
    avg_dropout_bad = np.array([j for j in data_lib_bad["dropout_0"]])
    dropout_norm_bad = np.array([1] * len(data_lib_bad["stack_0"]))

def make_some_features(data_lib,depth,num_features,img_size,avg_dropout,dropout_norm):
    for i in range(max_stack):
        for j in range(len(depth)):
            if cut and img_size[j] <= 1:
                data_lib["filters_"+str(2*i)][j] = 0
                data_lib["filters_"+str(2*i+1)][j] = 0
                data_lib["k_"+str(2*i)][j] = 0
                data_lib["k_"+str(2*i+1)][j] = 0
                data_lib["s_"+str(i)][j] = 0
                data_lib["stack_"+str(i)][j] = 0
            if (not cut) or img_size[j] > 1:
                depth[j] += data_lib["stack_"+str(i)][j]
                num_features[j] += data_lib["stack_"+str(i)][j] * data_lib["filters_" + str(2*i)][j]
                if data_lib["stack_"+str(i)][j] > 0:
                    avg_dropout[j] += data_lib["dropout_" + str(i+1)][j]
                    dropout_norm[j] += 1
                    img_size[j] = int(np.ceil(img_size[j] / data_lib["s_" + str(i)][j]))
                    if not data_lib["max_pooling"][j]:
                        num_features[j] += data_lib["filters_" + str(2*i+1)][j]
                        depth[j] += 1
    for j in range(len(depth)):
        if data_lib["dense_size_0"][j] > 0:
            avg_dropout[j] += data_lib["dropout_" + str(max_stack+1)][j]
            dropout_norm[j] += 1
        if data_lib["dense_size_1"][j] > 0:
            avg_dropout[j] += data_lib["dropout_" + str(max_stack+2)][j]
            dropout_norm[j] += 1
    if CIFAR10:
        for j in range(len(avg_dropout)):
            avg_dropout[j] /= dropout_norm[j]
    else:
        print("ERROR! first implement MNIST dropout normalisation")
        exit(0)
    return data_lib,depth,num_features,img_size,avg_dropout,dropout_norm

#CHRIS this is ugly, because numpy tends to make a new local variable "array" when assigning like array = array + bla, not necessarily passing "array" by reference
if not train_tweak:
    data_lib,depth,num_features,img_size,avg_dropout,dropout_norm = make_some_features(data_lib,depth,num_features,img_size,avg_dropout,dropout_norm)
    data_lib_good,depth_good,num_features_good,img_size_good,avg_dropout_good,dropout_norm_good = make_some_features(data_lib_good,depth_good,num_features_good,img_size_good,avg_dropout_good,dropout_norm_good)
    data_lib_bad,depth_bad,num_features_bad,img_size_bad,avg_dropout_bad,dropout_norm_bad = make_some_features(data_lib_bad,depth_bad,num_features_bad,img_size_bad,avg_dropout_bad,dropout_norm_bad)

def make_overlap(data_lib,img_size,overlap,total_skip,total_overlap,avg_skip_step,avg_skip_start,avg_kernel_size,avg_stride,avg_filters):
    avg_kernel_size_norm = np.array([0.0] * len(avg_kernel_size))
    avg_stride_norm = np.array([0.0] * len(avg_stride))
    avg_filters_norm = np.array([0.0]*len(avg_filters))
    for i in range(len(data_lib["stack_0"])):
        current_level = 1
        for j in range(max_stack):
            local_skip_start = [data_lib["skstart_"+str(l)][i] for l in range(5)]
            for k in range(data_lib["stack_"+str(j)][i]):
                test = [(current_level, local_skip_start[m], current_level - local_skip_start[m] , data_lib["skstep_"+str(m)][i]) for m in range(len(local_skip_start))]
                idx = sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 for m in range(len(local_skip_start))])
                idx -= sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and data_lib["skstep_"+str(n)][i] > 1 and current_level > local_skip_start[n] and m != n and data_lib["skstep_"+str(m)][i] == data_lib["skstep_"+str(n)][i] and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 and (current_level - local_skip_start[n]) % data_lib["skstep_"+str(n)][i] == 0 for m in range(len(local_skip_start)) for n in range(len(local_skip_start))])//2
                if idx > 0:
                    overlap[idx-1][i] += 1
                avg_kernel_size[i] += data_lib["k_"+str(j*2)][i]
                avg_kernel_size_norm[i] += 1
                avg_filters[i] += data_lib["filters_"+str(j*2)][i]
                avg_filters_norm[i] += 1
                current_level += 1
            if not data_lib["max_pooling"][i] and data_lib["stack_"+str(j)][i] > 0:
                test = [(current_level, local_skip_start[m], current_level - local_skip_start[m] , data_lib["skstep_"+str(m)][i]) for m in range(len(local_skip_start))]
                idx = sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 for m in range(len(local_skip_start))])
                idx -= sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and data_lib["skstep_"+str(n)][i] > 1 and current_level > local_skip_start[n] and m != n and data_lib["skstep_"+str(m)][i] == data_lib["skstep_"+str(n)][i] and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 and (current_level - local_skip_start[n]) % data_lib["skstep_"+str(n)][i] == 0 for m in range(len(local_skip_start)) for n in range(len(local_skip_start))])//2
                if idx > 0:
                    overlap[idx-1][i] += 1
                avg_kernel_size[i] += data_lib["k_"+str(j*2+1)][i]
                avg_kernel_size_norm[i] += 1
                avg_filters[i] += data_lib["filters_"+str(j*2+1)][i]
                avg_filters_norm[i] += 1
                current_level += 1
            if data_lib["stack_"+str(j)][i] > 0:
                avg_stride[i] += data_lib["s_"+str(j)][i]
                avg_stride_norm[i] += 1
                img_size[i] = int(np.ceil(img_size[i] / data_lib["s_" + str(j)][i]))
            if cut and img_size[i] <= 1:
                break
    for i in range(5):
        data_lib["overlap_"+str(i+1)]=overlap[i]
    for i in range(len(total_skip)):
        for j in range(5):
            total_skip[i] += (j+1) * overlap[j][i]
            total_overlap[i] += scipy.special.binom(j+1,2) * overlap[j][i]
    for i in range(len(avg_skip_step)):
        for j in range(5):
            avg_skip_step[i] += data_lib["skstep_" + str(j)][i]
        avg_skip_step[i] /= 5
    for i in range(len(avg_skip_start)):
        for j in range(5):
            avg_skip_start[i] += data_lib["skstart_" + str(j)][i]
        avg_skip_start[i] /= 5
    for i in range(len(avg_kernel_size_norm)):
        if avg_kernel_size_norm[i] == 0:
            avg_kernel_size_norm[i] = 1
        avg_kernel_size[i] /= avg_kernel_size_norm[i]
    for i in range(len(avg_stride_norm)):
        if avg_stride_norm[i] == 0:
            avg_stride_norm[i] = 1
        avg_stride[i] /= avg_stride_norm[i]
    for i in range(len(avg_filters_norm)):
        if avg_filters_norm[i] == 0:
            avg_filters_norm[i] = 1
        avg_filters[i] /= avg_filters_norm[i]
    return data_lib,img_size,overlap,total_skip,total_overlap,avg_skip_step,avg_skip_start,avg_kernel_size,avg_stride,avg_filters
            
if not train_tweak:
    img_size = np.array([img_dim] * len(data_lib["stack_0"]))
    overlap = np.array([[0]*len(data_lib["stack_0"])] * 5)
    total_skip = np.array([0] * len(data_lib["stack_0"]))
    total_overlap = np.array([0] * len(data_lib["stack_0"]))
    avg_skip_step = np.array([0.0] * len(data_lib["stack_0"]))
    avg_skip_start = np.array([0.0] * len(data_lib["stack_0"]))
    avg_kernel_size = np.array([0.0] * len(data_lib["stack_0"]))
    avg_stride = np.array([0.0] * len(data_lib["stack_0"]))
    avg_filters = np.array([0.0] * len(data_lib["stack_0"]))
    data_lib,img_size,overlap,total_skip,total_overlap,avg_skip_step,avg_skip_start,avg_kernel_size,avg_stride,avg_filters = make_overlap(data_lib,img_size,overlap,total_skip,total_overlap,avg_skip_step,avg_skip_start,avg_kernel_size,avg_stride,avg_filters)

    img_size_good = np.array([img_dim] * len(data_lib_good["stack_0"]))
    overlap_good = np.array([[0]*len(data_lib_good["stack_0"])] * 5)
    total_skip_good = np.array([0] * len(data_lib_good["stack_0"]))
    total_overlap_good = np.array([0] * len(data_lib_good["stack_0"]))
    avg_skip_step_good = np.array([0.0] * len(data_lib_good["stack_0"]))
    avg_skip_start_good = np.array([0.0] * len(data_lib_good["stack_0"]))
    avg_kernel_size_good = np.array([0.0] * len(data_lib_good["stack_0"]))
    avg_stride_good = np.array([0.0] * len(data_lib_good["stack_0"]))
    avg_filters_good = np.array([0.0] * len(data_lib_good["stack_0"]))
    data_lib_good,img_size_good,overlap_good,total_skip_good,total_overlap_good,avg_skip_step_good,avg_skip_start_good,avg_kernel_size_good,avg_stride_good,avg_filters_good = make_overlap(data_lib_good,img_size_good,overlap_good,total_skip_good,total_overlap_good,avg_skip_step_good,avg_skip_start_good,avg_kernel_size_good,avg_stride_good,avg_filters_good)


    img_size_bad = np.array([img_dim] * len(data_lib_bad["stack_0"]))
    overlap_bad = np.array([[0]*len(data_lib_bad["stack_0"])] * 5)
    total_skip_bad = np.array([0] * len(data_lib_bad["stack_0"]))
    total_overlap_bad = np.array([0] * len(data_lib_bad["stack_0"]))
    avg_skip_step_bad = np.array([0.0] * len(data_lib_bad["stack_0"]))
    avg_skip_start_bad = np.array([0.0] * len(data_lib_bad["stack_0"]))
    avg_kernel_size_bad = np.array([0.0] * len(data_lib_bad["stack_0"]))
    avg_stride_bad = np.array([0.0] * len(data_lib_bad["stack_0"]))
    avg_filters_bad = np.array([0.0] * len(data_lib_bad["stack_0"]))
    data_lib_bad,img_size_bad,overlap_bad,total_skip_bad,total_overlap_bad,avg_skip_step_bad,avg_skip_start_bad,avg_kernel_size_bad,avg_stride_bad,avg_filters_bad = make_overlap(data_lib_bad,img_size_bad,overlap_bad,total_skip_bad,total_overlap_bad,avg_skip_step_bad,avg_skip_start_bad,avg_kernel_size_bad,avg_stride_bad,avg_filters_bad)

    data_lib["avg_filters"] = avg_filters
    data_lib_good["avg_filters"] = avg_filters_good
    data_lib_bad["avg_filters"] = avg_filters_bad

    data_lib["avg_stride"] = avg_stride
    data_lib_good["avg_stride"] = avg_stride_good
    data_lib_bad["avg_stride"] = avg_stride_bad

    data_lib["avg_skip_step"] = avg_skip_step
    data_lib_good["avg_skip_step"] = avg_skip_step_good
    data_lib_bad["avg_skip_step"] = avg_skip_step_bad

    data_lib["avg_skip_start"] = avg_skip_start
    data_lib_good["avg_skip_start"] = avg_skip_start_good
    data_lib_bad["avg_skip_start"] = avg_skip_start_bad

    data_lib["avg_kernel_size"] = avg_kernel_size
    data_lib_good["avg_kernel_size"] = avg_kernel_size_good
    data_lib_bad["avg_kernel_size"] = avg_kernel_size_bad

    data_lib["total_skip"] = total_skip
    data_lib_good["total_skip"] = total_skip_good
    data_lib_bad["total_skip"] = total_skip_bad

    data_lib["total_overlap"] = total_overlap
    data_lib_good["total_overlap"] = total_overlap_good
    data_lib_bad["total_overlap"] = total_overlap_bad

    data_lib["depth"]=depth
    data_lib_good["depth"] = depth_good
    data_lib_bad["depth"] = depth_bad

    data_lib["num_features"] = num_features
    data_lib_good["num_features"] = num_features_good
    data_lib_bad["num_features"] = num_features_bad

    data_lib["avg_dropout"] = avg_dropout
    data_lib_good["avg_dropout"] = avg_dropout_good
    data_lib_bad["avg_dropout"] = avg_dropout_bad

data_panda = pd.DataFrame(data=data_lib)
data_panda_good = pd.DataFrame(data=data_lib_good)
data_panda_bad = pd.DataFrame(data=data_lib_bad)

def normalize_panda_data(data_panda):
    select = [x for x in data_panda.columns if x != "time" and x != "acc" and x != "activation" and x != "activ_dense" and x != "fill_mode"]
    selection = data_panda.loc[:, select]
    normalizer = selection.max()-selection.min()
    for i in range(normalizer.shape[0]):
        if normalizer[i] == 0:
            if selection.max()[i] == 0:
                normalizer[i] = 1.0
            else:
                normalizer[i] = selection.max()[i]

    normalized_df=(selection-selection.min())/normalizer
    return normalized_df

def normalize_two_panda_data(data_panda_1, data_panda_2):
    select_1 = [x for x in data_panda_1.columns if x != "activation" and x != "activ_dense" and x != "fill_mode"]
    select_2 = [x for x in data_panda_2.columns if x != "activation" and x != "activ_dense" and x != "fill_mode"]
    selection_1 = data_panda_1.loc[:, select_1]
    selection_2 = data_panda_2.loc[:, select_2]
    selection_1["good"] = [True] * len(selection_1[selection_1.columns.values[0]])
    selection_2["good"] = [False] * len(selection_2[selection_2.columns.values[0]])
    combo = pd.concat([selection_1, selection_2])
    maxert = combo.max()
    minnert = combo.min()
    normalizer = maxert-minnert
    for i in range(normalizer.shape[0]):
        if normalizer[i] == 0:
            if maxert[i] == 0:
                normalizer[i] = 1.0
            else:
                normalizer[i] = combo.max()[i]
    
    normalized_combo = (combo -minnert)/normalizer
    normalized_df_good = (selection_1 -minnert)/normalizer
    normalized_df_good=normalized_df_good.drop(columns="good")
    normalized_df_bad = (selection_2 -minnert)/normalizer
    normalized_df_bad=normalized_df_bad.drop(columns="good")
    return normalized_combo,normalized_df_good,normalized_df_bad

strides = ["s_"+str(i) for i in range(max_stack)]
#select = [x for x in data_panda.columns if x == 'avg_dropout' or x == 'avg_kernel_size' or x == 'num_features' or x == 'time' or x == 'l2' or x == 'dropout_0' or x == 'elu' or x == 'batch_size_sp' or x == 'epoch_sp' or x == 'lr' or x == 'max_pooling']
#select = [x for x in data_panda.columns if x == 'avg_dropout' or x == 'avg_kernel_size' or x == 'num_features' or x == 'time' or x == 'l2' or x == 'dropout_0' or x == 'elu' or x == 'batch_size_sp' or x == 'epoch_sp' or x == 'lr' or x == 'max_pooling' or x == 'channel_shift_range' or x == 'constant' or x == 'global_pooling' or x == 'height_shift_range' or x == 'horizontal_flip' or x == 'max_pooling' or x == 'nearest' or x == 'rotation_range' or x == 'vertical_flip' or x == 'width_shift_range' or x == 'zoom_range']
#select = [x for x in data_panda.columns if x == 'lr' or x == 'drop' or x == 'epochs_drop' or x == 'momentum' or x == 'SGD' or x == 'RMSprop' or x == 'Adagrad' or x == 'Adadelta' or x == 'Adam' or x == 'Adamax' or x == 'Nadam' or x == 'rho']
select = [x for x in data_panda.columns if x == 'time']
#select = [x for x in data_panda.columns if any(x == m for m in strides)]
#select = [x for x in data_panda.columns]
#normalized_df= normalize_panda_data(data_panda.loc[:,select])
normalized_df= data_panda.loc[:,select]
#normalized_df_combo,normalized_df_good,normalized_df_bad= normalize_two_panda_data(data_panda_good.loc[:,select],data_panda_bad.loc[:,select])
normalized_df_good,normalized_df_bad= data_panda_good.loc[:,select],data_panda_bad.loc[:,select]

if do_parallel_plot:
    #color_good = {'boxes': 'DarkGreen', 'whiskers': 'DarkGreen','medians': 'DarkGreen', 'caps': 'Green'}
    #normalized_df_good.plot.box(color=color_good, sym='g+')
    #color_bad = {'boxes': 'DarkRed', 'whiskers': 'DarkRed','medians': 'DarkRed', 'caps': 'Red'}
    #normalized_df_combo.plot.box(by="good")
    boxprops_good = dict(linestyle='-', linewidth=4, color='green')
    medianprops_good = dict(linestyle='-', linewidth=4, color='green')

    fig = matplotlib.pyplot.gcf()
    ax = normalized_df_good.plot(kind='box',
            color=dict(boxes='g', whiskers='g', medians='g', caps='g'),
            boxprops=dict(linestyle='-', linewidth=2.0),
            flierprops=dict(linestyle='-', linewidth=2.0),
            medianprops=dict(linestyle='-', linewidth=2.0),
            whiskerprops=dict(linestyle='-', linewidth=2.0),
            capprops=dict(linestyle='-', linewidth=2.0),
            showfliers=False, grid=True, rot=0)
    normalized_df_bad.plot(kind='box',
             color=dict(boxes='r', whiskers='r', medians='r', caps='r'),
             boxprops=dict(linestyle='-', linewidth=1.0),
             flierprops=dict(linestyle='-', linewidth=1.0),
             medianprops=dict(linestyle='-', linewidth=1.0),
             whiskerprops=dict(linestyle='-', linewidth=1.0),
             capprops=dict(linestyle='-', linewidth=1.0),
             showfliers=False, grid=True, rot=0,ax=ax)
    plt.show()
    #sns.boxplot(x="variable", y="value",data=pd.melt(normalized_df_combo))#,hue="good", palette="Set3"
    #for i in range(normalized_df.shape[0]):
    #    #parallel_coordinates(pd.DataFrame(data=normalized_df.loc[i:i]),'acc',alpha=normalized_df.loc[i:i]['acc'].values)#, colormap=plt.get_cmap("Set2"))
    #fig.set_size_inches(180, 105)
    #fig.savefig('parallel_coord_plot_cut.png',dpi=100)
    #fig.savefig('parallel_coord_plot_cut.png')

def forbidden(i,j):#Filters out correlations that are too obvious
    if (i == 'total_overlap' and 'overlap' in j) or (j == 'total_overlap' and 'overlap' in i):
        return True
    if (i == 'total_skip' and j == 'depth') or (j == 'total_skip' and i == 'depth'):
        return True
    if ('overlap' in i and j == 'total_skip') or ('overlap' in j and i == 'total_skip'):
        return True
    if ('overlap' in i and j == 'depth') or ('overlap' in j and i == 'depth'):
        return True
    if ('overlap' in i and j == 'num_features') or ('overlap' in j and i == 'num_features'):
        return True
    if (i == 'depth' and j == 'num_features') or (j == 'depth' and i == 'num_features'):
        return True
    if (i == 'avg_dropout' and  'dropout' in j) or (j == 'avg_dropout' and  'dropout' in i):
        return True
    if (i == 'total_skip' and j == 'max_pooling') or (j == 'total_skip' and i == 'max_pooling'):
        return True
    if (i == 'max_pooling' and j == 'depth') or (j == 'max_pooling' and i == 'depth'):
        return True
    if ('overlap' in i and j == 'max_pooling') or ('overlap' in j and i == 'max_pooling'):
        return True
    if (i == 'avg_skip_step' and 'skstep_' in j) or (j == 'avg_skip_step' and 'skstep' in i):
        return True
    if (i == 'avg_skip_start' and 'skstart_' in j) or (j == 'avg_skip_start' and 'skstart' in i):
        return True
    if (i == 'avg_filters' and 'filters_' in j) or (j == 'avg_filters' and 'filters_' in i):
        return True
    #begin for tweaked:
    if (i == 'avg_filters' and j == 'num_features') or (j == 'avg_filters' and i == 'num_features'):
        return True
    if (i == 'depth' and 'stack_' in j) or (j == 'depth' and 'stack_' in i):
        return True
    if (i == 'total_skip' and 'stack_' in j) or (j == 'total_skip' and 'stack_' in i):
        return True
    if (i == 'num_features' and j == 'total_skip') or (j == 'num_features' and i == 'total_skip'):
        return True
    if ('overlap_' in i and 'stack_' in j) or ('overlap_' in j and 'stack_' in i):
        return True
    if (i == 'num_features' and 'stack_' in j) or (j == 'num_features' and 'stack_' in i):
        return True
    if (i == 'avg_skip_step' and 'overlap_' in j) or (j == 'avg_skip_step' and 'overlap_' in i):
        return True
    if (i == 'time' and 'filters_' in j) or (j == 'time' and 'filters_' in i):
        return True
    if (i == 'avg_skip_step' and j == 'total_skip') or (j == 'avg_skip_step' and i == 'total_skip'):
        return True
    if (i == 'avg_skip_step' and j == 'total_overlap') or (j == 'avg_skip_step' and i == 'total_overlap'):
        return True
    if ('skstep_' in i and 'overlap_' in j) or ('skstep_' in j and 'overlap_' in i):
        return True
    if (i == 'total_overlap' and 'stack_' in j) or (j == 'total_overlap' and 'stack_' in i):
        return True
    if (i == 'max_pooling' and j == 'num_features') or (j == 'max_pooling' and i == 'num_features'):
        return True
    #end for tweaked
    if ('overlap_' in i and 'overlap_' in j):
        return True
    kernels = ["k_"+str(i) for i in range(2*max_stack)]
    if any((i == 'avg_kernel_size' and j == m) or (j == 'avg_kernel_size' and i == m) for m in kernels):
        return True
    strides = ["s_"+str(i) for i in range(max_stack)]
    if any((i == 'avg_stride' and j == m) or (j == 'avg_stride' and i == m) for m in strides):
        return True
    act_funcs = ["elu","relu","tanh","sigmoid","selu"]
    if any(i == m and j == n for m in act_funcs for n in act_funcs):
        return True
    filters = ["filters_"+str(i) for i in range(2*max_stack)]
    stacks =  ["stack_"+str(i) for i in range(max_stack)]
    forbidden_set = [kernels,strides,filters,stacks]
    if cut:
        for v in forbidden_set:
            for z in forbidden_set:
                if any(i == m and j == n for m in v for n in z):
                    return True
    return False

def give_top_labels(selection, top, plot=False,save_name='test',compare_good_bad=False,bad=None):
    correlations = selection.corr()
    idx_pairs = [(i,j) for i in correlations for j in correlations if i != j and (not math.isnan(correlations.loc[i,j])) and (not forbidden(i,j))]
    #print([abs(correlations.loc[idx_pairs[i][0],idx_pairs[i][1]]) for i in range(len(idx_pairs))])
    sorted_idx_pairs = np.argsort([abs(correlations.loc[idx_pairs[i][0],idx_pairs[i][1]]) for i in range(len(idx_pairs))])[::-1]
    #print(sorted_idx_pairs)
    corr_pivot_labels = set([])
    i = 0
    total = 0
    while i < len(sorted_idx_pairs) and total < top:
        if not compare_good_bad and plot and i % 2 == 0:
            x=[x for x in selection[idx_pairs[sorted_idx_pairs[i]][0]]]
            y=[y for y in selection[idx_pairs[sorted_idx_pairs[i]][1]]]
            plt.clf()
            plt.cla()
            plt.xlabel(idx_pairs[sorted_idx_pairs[i]][0]+'            correlation = ' + str(round(correlations.loc[idx_pairs[sorted_idx_pairs[i]][0]][idx_pairs[sorted_idx_pairs[i]][1]],2)))
            plt.ylabel(idx_pairs[sorted_idx_pairs[i]][1])
            sns.kdeplot(x,y,cmap="Blues_d")
            plt.savefig(save_name + '_n_'+ str(i//2) +'.png')
        elif compare_good_bad and i % 2 == 0:
            x=[x for x in selection[idx_pairs[sorted_idx_pairs[i]][0]]]
            y=[y for y in selection[idx_pairs[sorted_idx_pairs[i]][1]]]
            plt.clf()
            plt.cla()
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2,sharex=ax1,sharey=ax1)
            ax1.set(xlabel=idx_pairs[sorted_idx_pairs[i]][0]+' good corr = ' + str(round(correlations.loc[idx_pairs[sorted_idx_pairs[i]][0]][idx_pairs[sorted_idx_pairs[i]][1]],2)), ylabel=idx_pairs[sorted_idx_pairs[i]][1])
            #plt.xlabel(idx_pairs[sorted_idx_pairs[i]][0]+' correlation = ' + str(round(correlations.loc[idx_pairs[sorted_idx_pairs[i]][0]][idx_pairs[sorted_idx_pairs[i]][1]],2)))
            #plt.ylabel(idx_pairs[sorted_idx_pairs[i]][1])
            try:
                sns.kdeplot(x,y,cmap="Blues_d",ax=ax1)
            except:
                print("Could not plot good for " + str(idx_pairs[sorted_idx_pairs[i]][0]) + " and " + str(idx_pairs[sorted_idx_pairs[i]][1]))
            correlations_bad = bad.corr()
            x=[x for x in bad[idx_pairs[sorted_idx_pairs[i]][0]]]
            y=[y for y in bad[idx_pairs[sorted_idx_pairs[i]][1]]]
            ax2.set(xlabel=idx_pairs[sorted_idx_pairs[i]][0]+' bad corr = ' + str(round(correlations_bad.loc[idx_pairs[sorted_idx_pairs[i]][0]][idx_pairs[sorted_idx_pairs[i]][1]],2)), ylabel=idx_pairs[sorted_idx_pairs[i]][1])
            #plt.xlabel(idx_pairs[sorted_idx_pairs[i]][0]+'            correlation = ' + str(round(correlations.loc[idx_pairs[sorted_idx_pairs[i]][0]][idx_pairs[sorted_idx_pairs[i]][1]],2)))
            #plt.ylabel(idx_pairs[sorted_idx_pairs[i]][1])
            try:
                sns.kdeplot(x,y,cmap="Blues_d",ax=ax2)
            except:
                print("Could not plot bad for " + str(idx_pairs[sorted_idx_pairs[i]][0]) + " and " + str(idx_pairs[sorted_idx_pairs[i]][1]))
            plt.savefig(save_name + '_n_'+ str(i//2) +'.png')
        plt.close("all")
        print(idx_pairs[sorted_idx_pairs[i]][0],idx_pairs[sorted_idx_pairs[i]][1])
        print(correlations.loc[idx_pairs[sorted_idx_pairs[i]][0],idx_pairs[sorted_idx_pairs[i]][1]])
        corr_pivot_labels = corr_pivot_labels.union(set([idx_pairs[sorted_idx_pairs[i]][0],idx_pairs[sorted_idx_pairs[i]][1]]))
        total += 1
        i+=1
                
    return list(corr_pivot_labels)

if do_pairgrid:
    def make_pairgrid(data_panda,save_name,top):
        select = [x for x in data_panda.columns]
        selection = data_panda.loc[:, select]
        corr_pivot_labels = give_top_labels(selection,top,plot=True,save_name=save_name)
        #g = sns.PairGrid(data_panda_good,vars=['acc','time','stack_0','stack_1','stack_2','stack_3','stack_4','stack_5','stack_6','s_0','s_1','s_2','s_3','s_4','s_5','s_6','filters_0','filters_1','filters_2','filters_3','filters_4','filters_5','filters_6','filters_7','filters_8','filters_9','filters_10','filters_11','filters_12','filters_13','k_0','k_1','k_2','k_3','k_4','k_5','k_6','k_7','k_8','k_9','k_10','k_11','k_12','k_13','dropout_0','dropout_1','dropout_2','dropout_3','dropout_4','dropout_5','dropout_6','dropout_7','dropout_8','dropout_9','lr','l2','global_pooling','skstart_0','skstart_1','skstart_2','skstart_3','skstart_4','skstep_0','skstep_1','skstep_2','skstep_3','skstep_4','dense_size_0','dense_size_1','epoch_sp','batch_size_sp'],hue='acc',palette='GnBu_d')
        #g = sns.PairGrid(data_panda,vars=['acc','time','lr','l2','s_0','s_1','filters_0','filters_1','k_0','k_1','dropout_0','dropout_1','dense_size_0','dense_size_1'],hue='acc',palette='GnBu_d')
        #g = sns.PairGrid(data_panda,vars=['filters_0','filters_1','filters_2','filters_3','filters_4','filters_5','filters_6','filters_7','filters_8','filters_9','filters_10','filters_11','filters_12','filters_13'],hue='acc',palette='GnBu_d')
        #g = sns.PairGrid(data_panda,vars=['k_0','k_1','k_2','k_3','k_4','k_5','k_6','k_7','k_8','k_9','k_10','k_11','k_12','k_13'],hue='acc',palette='GnBu_d')
        #g = sns.PairGrid(data_panda,vars=["softmax","elu","relu","tanh","sigmoid","selu"],hue='acc',palette='GnBu_d')
        #g = sns.PairGrid(data_panda,vars=["acc","time","epoch_sp","batch_size_sp","depth"],hue='acc',palette='GnBu_d')
        plt.clf()
        plt.cla()
        #g = sns.PairGrid(data_panda,vars=corr_pivot_labels)
        #g = sns.PairGrid(data_panda,vars=corr_pivot_labels,hue='acc',palette='GnBu_d')
        #g.map(plt.scatter)
        #g.map_upper(plt.scatter)
        #g.map_lower(sns.kdeplot, cmap="Blues_d")
        #g.map_diag(sns.kdeplot, lw=3, legend=False)
        #g.map_diag(plt.scatter)

        #plt.savefig('load_analysis_output.png')
        #plt.savefig(save_name+'.png')
    top = 80
    give_top_labels(data_panda_good,top,plot=True,save_name='load_analysis_output_test_compare',compare_good_bad=True,bad=data_panda_bad)
    print()
    print('top ' + str(top) + ' correlations on all data')
    print()
    #make_pairgrid(data_panda,'load_analysis_output_test',top)
    print()
    print('top ' + str(top) + ' correlations on good data')
    print()
    #make_pairgrid(data_panda_good,'load_analysis_output_test_good',top)
    print()
    print('top ' + str(top) + ' correlations on bad data')
    print()
    #make_pairgrid(data_panda_bad,'load_analysis_output_test_bad',top)

if do_correlations:
    corr_pivot = 0.3
    #parallel_coordinates(data_panda, class_column='acc', cols=['lr','l2'])
    select = [x for x in data_panda.columns]
    #select = [x for x in data_panda.columns if x == "acc" or x == "time" or x == "depth" or x == "epoch_sp" or x == "batch_size_sp"]
    selection = data_panda.loc[:, select]
    correlations = selection.corr()
    top = 20
    
    print()
    print('top ' + str(top) + ' correlations on all data')
    print()
    corr_pivot_labels = give_top_labels(selection, top)
    corr_select = selection[corr_pivot_labels].corr()
    #plt.matshow(correlations)
    sns.heatmap(corr_select,xticklabels=corr_select.columns.values,yticklabels=corr_select.columns.values,cmap='coolwarm')
    #correlations.style.background_gradient(cmap='coolwarm').set_precision(2)
    plt.show()
    
    select_good = [x for x in data_panda_good.columns]
    selection_good = data_panda_good.loc[:, select_good]
    correlations_good = selection_good.corr()
    
    print()
    print('top ' + str(top) + ' correlations on good data')
    print()
    corr_pivot_labels_good = give_top_labels(selection_good, top)
    corr_select_good = selection_good[corr_pivot_labels_good].corr()
    sns.heatmap(corr_select_good,xticklabels=corr_select_good.columns.values,yticklabels=corr_select_good.columns.values,cmap='coolwarm')
    plt.show()

    select_bad = [x for x in data_panda_bad.columns]
    selection_bad = data_panda_bad.loc[:, select_bad]
    correlations_bad = selection_bad.corr()
    
    print()
    print('top ' + str(top) + ' correlations on bad data')
    print()
    corr_pivot_labels_bad = give_top_labels(selection_bad, top)
    corr_select_bad = selection_bad[corr_pivot_labels_bad].corr()
    sns.heatmap(corr_select_bad,xticklabels=corr_select_bad.columns.values,yticklabels=corr_select_bad.columns.values,cmap='coolwarm')
    plt.show()

select = [x for x in data_panda.columns if x != "time" and x != "acc" and x != "activation" and x != "activ_dense"]
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(data_panda.loc[:, select])

if do_k_means:
    # K Means Cluster
    data_panda_real = data_panda.loc[:, select]
    model_kmeans = KMeans(n_clusters=2)
    model_kmeans.fit(scaler.transform(data_panda_real))
    # Create a colormap
    colormap = np.array(['red', 'lime', 'black','blue'])
    plt.clf()
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.scatter(data_panda.time,data_panda.acc, c=colormap[model_kmeans.labels_], s=1)
    plt.savefig('Kmeans_2_cut.png')

if do_dbscan:
    n_clusters = 0
    for i in range(1,100):
        for j in range(1,100):
            #model = DBSCAN(eps=10, min_samples=6)# (8,2)
            model = DBSCAN(eps=i, min_samples=j)
            model.fit(scaler.transform(data_panda.loc[:, select]))
            #print('number of clusters:')
            n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
            #print(n_clusters)
            if n_clusters >= 2 and n_clusters <= 10:
                print(i,j)
                plt.clf()
                plt.xlabel('time')
                plt.ylabel('accuracy')
                plt.scatter(data_panda.time,data_panda.acc, c=model.labels_, s=1)
                plt.savefig('DBSCAN_eps_' + str(i) + '_mins_' + str(j) + '_cut.png')
    plt.scatter(data_panda.time,data_panda.acc, c=model.labels_, s=1)

    #plt.show()

if do_rule_finding:
    n_sections = 10

    string_cast_dict = {}
    max_val_dict = {}
    min_val_dict = {}
    
    select = [x for x in data_panda.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu" and x != "overlap_5" and x != "overlap_4" and x != "stack_0" and x != "stack_1" and x != "stack_2" and x != "stack_3" and x != "stack_4" and x != "stack_5" and x != "stack_6" and x!= "k_0" and x!= "k_1" and x!= "k_2" and x!= "k_3" and x!= "k_4" and x!= "k_5" and x!= "k_6" and x!= "k_7" and x!= "k_8" and x!= "k_9" and x!= "k_10" and x!= "k_11" and x!= "k_12" and x!= "k_13" and x!= "filters_0" and x!= "filters_1" and x!= "filters_2" and x!= "filters_3" and x!= "filters_4" and x!= "filters_5" and x!= "filters_6" and x!= "filters_7" and x!= "filters_8" and x!= "filters_9" and x!= "filters_10" and x!= "filters_11" and x!= "filters_12" and x!= "filters_13" and x!= "s_0" and x!= "s_1" and x!= "s_2" and x!= "s_3" and x!= "s_4" and x!= "s_5" and x!= "s_6" and x!= "s_7" and x!= "s_8" and x!= "s_9" and x!= "s_10" and x!= "s_11" and x!= "s_12" and x!= "s_13"]
    selection = data_panda.loc[:, select]

    data_discrete = []
    first = True
    for x in selection.columns:
        if x != "activation" and x != "activ_dense":
            max_val = float(max(selection[x]))
            min_val = float(min(selection[x]))
            string_cast = [x + ': ' + str(s * (max_val-min_val)/n_sections + min_val) + ' - ' + str((s+1) * (max_val-min_val)/n_sections + min_val) for s in range(n_sections)]
            string_cast_dict[x] = string_cast
            max_val_dict[x] = max_val
            min_val_dict[x] = min_val
            #print(string_cast)
            if first:
                if max_val - min_val == 0:
                    for d in selection[x]:
                        data_discrete.append([string_cast[min(n_sections-1,int((float(d) - min_val)*n_sections))]])
                else:
                    for d in selection[x]:
                        data_discrete.append([string_cast[min(n_sections-1,int((float(d) - min_val)/(max_val-min_val)*n_sections))]])
            else:
                if max_val - min_val == 0:
                    for i in range(len(selection[x])):
                        data_discrete[i].append(string_cast[min(n_sections-1,int((float(selection[x][i]) - min_val)*n_sections))])
                else:
                    for i in range(len(selection[x])):
                        data_discrete[i].append(string_cast[min(n_sections-1,int((float(selection[x][i]) - min_val)/(max_val-min_val)*n_sections))])
        else:
            if first:
                for d in selection[x]:
                    data_discrete.append([d])
            else:
                for i in range(len(selection[x])):
                    data_discrete[i].append(data_panda[x])
        first = False
    
    for i in range(len(data_discrete)):
        data_discrete[i] = set(data_discrete[i])
    
    select_good = [x for x in data_panda_good.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu" and x != "overlap_5" and x != "overlap_4" and x != "stack_0" and x != "stack_1" and x != "stack_2" and x != "stack_3" and x != "stack_4" and x != "stack_5" and x != "stack_6" and x!= "k_0" and x!= "k_1" and x!= "k_2" and x!= "k_3" and x!= "k_4" and x!= "k_5" and x!= "k_6" and x!= "k_7" and x!= "k_8" and x!= "k_9" and x!= "k_10" and x!= "k_11" and x!= "k_12" and x!= "k_13" and x!= "filters_0" and x!= "filters_1" and x!= "filters_2" and x!= "filters_3" and x!= "filters_4" and x!= "filters_5" and x!= "filters_6" and x!= "filters_7" and x!= "filters_8" and x!= "filters_9" and x!= "filters_10" and x!= "filters_11" and x!= "filters_12" and x!= "filters_13" and x!= "s_0" and x!= "s_1" and x!= "s_2" and x!= "s_3" and x!= "s_4" and x!= "s_5" and x!= "s_6" and x!= "s_7" and x!= "s_8" and x!= "s_9" and x!= "s_10" and x!= "s_11" and x!= "s_12" and x!= "s_13"]
    selection_good = data_panda_good.loc[:, select_good]
    
    data_discrete_good = []
    first = True
    for x in selection_good.columns:
        if x != "activation" and x != "activ_dense":
            if first:
                if max_val_dict[x]-min_val_dict[x] == 0:
                    for d in selection_good[x]:
                        data_discrete_good.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])*n_sections))]])
                else:
                    for d in selection_good[x]:
                        data_discrete_good.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))]])
            else:
                if max_val_dict[x]-min_val_dict[x] == 0:
                    for i in range(len(selection_good[x])):
                        data_discrete_good[i].append(string_cast_dict[x][min(n_sections-1,int((float(selection_good[x][i]) - min_val_dict[x])*n_sections))])
                else:
                    for i in range(len(selection_good[x])):
                        data_discrete_good[i].append(string_cast_dict[x][min(n_sections-1,int((float(selection_good[x][i]) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))])
        else:
            if first:
                for d in selection_good[x]:
                    data_discrete_good.append([d])
            else:
                for i in range(len(selection_good[x])):
                    data_discrete_good[i].append(selection_good[x])
        first = False

    for i in range(len(data_discrete_good)):
        data_discrete_good[i] = set(data_discrete_good[i])
    
    select_bad = [x for x in data_panda_bad.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu" and x != "overlap_5" and x != "overlap_4" and x != "stack_0" and x != "stack_1" and x != "stack_2" and x != "stack_3" and x != "stack_4" and x != "stack_5" and x != "stack_6" and x!= "k_0" and x!= "k_1" and x!= "k_2" and x!= "k_3" and x!= "k_4" and x!= "k_5" and x!= "k_6" and x!= "k_7" and x!= "k_8" and x!= "k_9" and x!= "k_10" and x!= "k_11" and x!= "k_12" and x!= "k_13" and x!= "filters_0" and x!= "filters_1" and x!= "filters_2" and x!= "filters_3" and x!= "filters_4" and x!= "filters_5" and x!= "filters_6" and x!= "filters_7" and x!= "filters_8" and x!= "filters_9" and x!= "filters_10" and x!= "filters_11" and x!= "filters_12" and x!= "filters_13" and x!= "s_0" and x!= "s_1" and x!= "s_2" and x!= "s_3" and x!= "s_4" and x!= "s_5" and x!= "s_6" and x!= "s_7" and x!= "s_8" and x!= "s_9" and x!= "s_10" and x!= "s_11" and x!= "s_12" and x!= "s_13"]
    selection_bad = data_panda_bad.loc[:, select_bad]

    data_discrete_bad = []
    first = True
    for x in selection_bad.columns:
        if x != "activation" and x != "activ_dense":
            if first:
                if max_val_dict[x]-min_val_dict[x] == 0:
                    for d in selection_bad[x]:
                        data_discrete_bad.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])*n_sections))]])
                else:
                    for d in selection_bad[x]:
                        data_discrete_bad.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))]])
            else:
                if max_val_dict[x]-min_val_dict[x] == 0:
                    for i in range(len(selection_bad[x])):
                        data_discrete_bad[i].append(string_cast_dict[x][min(n_sections-1,int((float(selection_bad[x][i]) - min_val_dict[x])*n_sections))])
                else:
                    for i in range(len(selection_bad[x])):
                        data_discrete_bad[i].append(string_cast_dict[x][min(n_sections-1,int((float(selection_bad[x][i]) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))])
        else:
            if first:
                for d in selection_bad[x]:
                    data_discrete_bad.append([d])
            else:
                for i in range(len(selection_bad[x])):
                    data_discrete_bad[i].append(selection_bad[x])
        first = False
        
    for i in range(len(data_discrete_bad)):
        data_discrete_bad[i] = set(data_discrete_bad[i])

    #print(data_discrete)
    
    #example of possible function parameters
    #apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
    min_support = 0.5
    results = list(apriori(data_discrete,min_support=min_support))
    
    patterns = []
    print('\n1 items or more top 10 support:\n')
    for i in results:
        if len(i.items) >= 1:
            patterns.append(i)
    idx_sort = np.argsort([i.support for i in patterns])[::-1]
    for i in range(min(len(idx_sort),10)):
        print()
        print(patterns[idx_sort[i]].items)
        print("support in all data:")
        print(patterns[idx_sort[i]].support)

    #print(len(data_discrete))
    #print(len(data_discrete_good))
    #print(len(data_discrete_bad))
    
    min_support = 0.5
    results_good = list(apriori(data_discrete_good,min_support=min_support))
    min_support = 0.5
    results_bad = list(apriori(data_discrete_bad,min_support=min_support))
    
    good_patterns = []
    print('\ngood, 1 items or more top 10 support:\n')
    for i in results_good:
        if len(i.items) >= 1:
            good_patterns.append(i)
    idx_sort = np.argsort([i.support for i in good_patterns])[::-1]
    for i in range(min(len(idx_sort),10)):
        print()
        print(good_patterns[idx_sort[i]].items)
        print("support in good:")
        print(good_patterns[idx_sort[i]].support)

    bad_patterns = []
    print('\nbad, 1 items or more top 10 support:\n')
    for i in results_bad:
        if len(i.items) >= 1:
            bad_patterns.append(i)
    idx_sort = np.argsort([i.support for i in bad_patterns])[::-1]
    for i in range(min(len(idx_sort),10)):
        print()
        print(bad_patterns[idx_sort[i]].items)
        print("support in bad:")
        print(bad_patterns[idx_sort[i]].support)
    
    good_patterns_not_in_bad = []
    both_good = []
    both_bad = []
    print('\ngood patterns less in bad (1 items or more) top 10:\n')
    for i in results_good:
        exist = False
        for j in results_bad:
            if i.items == j.items:
                exist = True
                if i.support > j.support:
                    both_good.append(i)
                    both_bad.append(j)
        if not exist:
            good_patterns_not_in_bad.append(i)
    idx_sort = np.argsort([i.support for i in good_patterns_not_in_bad])[::-1]
    for i in range(min(len(idx_sort),10)):
        print()
        print(good_patterns_not_in_bad[idx_sort[i]].items)
        print("support in good:")
        print(good_patterns_not_in_bad[idx_sort[i]].support)
        print("support in bad:")
        print("< " + str(min_support))
    both_idx = np.argsort([both_good[i].support/both_bad[i].support for i in range(len(both_good))])[::-1]
    for i in range(max(0,min(10-len(idx_sort),len(both_idx)))):
        print()
        print(both_good[both_idx[i]].items)
        print("support in good:")
        print(both_good[both_idx[i]].support)
        print("support in bad:")
        print(both_bad[both_idx[i]].support)
    
    bad_patterns_not_in_good = []
    both_good = []
    both_bad = []
    print('\nbad patterns less in good (1 items or more) top 10:\n')
    for i in results_bad:
        exist = False
        for j in results_good:
            if i.items == j.items:
                exist = True
                if i.support > j.support:
                    both_good.append(j)
                    both_bad.append(i)
        if not exist:
            bad_patterns_not_in_good.append(i)
    idx_sort = np.argsort([i.support for i in bad_patterns_not_in_good])[::-1]
    for i in range(min(len(idx_sort),10)):
        print()
        print(bad_patterns_not_in_good[idx_sort[i]].items)
        print("support in good:")
        print("< " + str(min_support))
        print("support in bad:")
        print(bad_patterns_not_in_good[idx_sort[i]].support)
    both_idx = np.argsort([both_bad[i].support/both_good[i].support for i in range(len(both_good))])[::-1]
    for i in range(max(0,min(10-len(idx_sort),len(both_idx)))):
        print()
        print(both_good[both_idx[i]].items)
        print("support in good:")
        print(both_good[both_idx[i]].support)
        print("support in bad:")
        print(both_bad[both_idx[i]].support)
    
    union_patterns = []
    bad_union_support = []
    print('\nunion, 1 items or more top 10 most difference:\n')
    for i in results_good:
        if len(i.items) >= 1:
            for j in results_bad:
                if len(j.items) >= 1 and i.items == j.items:
                    union_patterns.append(i)
                    bad_union_support.append(j.support)
    top_diff_idx = np.argsort([union_patterns[i].support / bad_union_support[i] for i in range(len(union_patterns))])[::-1]
    for i in range(min(10,len(top_diff_idx))):
        print()
        print(union_patterns[top_diff_idx[i]].items)
        print("support in good:")
        print(union_patterns[top_diff_idx[i]].support)
        print("support in bad:")
        print(bad_union_support[top_diff_idx[i]])
    
    p_union_s_good = []
    for p in union_patterns:
        p_union_s_good_append = []
        for d in data_discrete_good:
            if p.items.issubset(d):
                p_union_s_good_append.append(d.difference(p.items))
        p_union_s_good.append(p_union_s_good_append)
    
    p_union_s_bad = []
    for p in union_patterns:
        p_union_s_bad_append = []
        for d in data_discrete_bad:
            if p.items.issubset(d):
                p_union_s_bad_append.append(d.difference(p.items))
        p_union_s_bad.append(p_union_s_bad_append)
    
    print("\nrules:")
    min_support = 0.5
    p_union_s_good_results = []
    p_union_s_bad_results = []
    for i in range(len(union_patterns)):
        p_union_s_good_results.append(list(apriori(p_union_s_good[i],min_support=min_support)))
        p_union_s_bad_results.append(list(apriori(p_union_s_bad[i],min_support=min_support)))
    
    top_good_idx = np.argsort([i.support for i in union_patterns])[::-1]
    top_bad_idx = np.argsort([i for i in bad_union_support])[::-1]
    
    for i in range(min(len(top_diff_idx),10)):
        #idx = top_good_idx[i]
        idx = top_diff_idx[i]
        #if i < 5:
        #    idx = top_good_idx[i]
        #else:
        #    idx = top_bad_idx[i-5]
        print()
        print("If:")
        print(union_patterns[idx].items)
        print("good support:")
        print(union_patterns[idx].support)
        print("bad support:")
        print(bad_union_support[idx])
        print()
        print("-->")
        print()
        print("do:")
        print()
        do = []
        maybe_do_good = []
        maybe_do_bad = []
        for j in range(len(p_union_s_good_results[idx])):
            exist_equal = False
            for k in range(len(p_union_s_bad_results[idx])):
                if p_union_s_good_results[idx][j].items == p_union_s_bad_results[idx][k].items:
                    exist_equal = True
                    if p_union_s_good_results[idx][j].support > p_union_s_bad_results[idx][k].support:
                        maybe_do_good.append(p_union_s_good_results[idx][j])
                        maybe_do_bad.append(p_union_s_bad_results[idx][k])
            if not exist_equal and not any('acc:' in string for string in p_union_s_good_results[idx][j].items) and not any('time:' in string for string in p_union_s_good_results[idx][j].items):
                do.append(p_union_s_good_results[idx][j])
        do_idx = np.argsort([j.support for j in do])[::-1]
        for j in range(min(5,len(do_idx))):
            print()
            print(do[do_idx[j]].items)
            print("support in good subset selection:")
            print(do[do_idx[j]].support)
            print("support in bad subset selection:")
            print("< " + str(min_support))
        maybe_do_idx = np.argsort([maybe_do_good[j].support /maybe_do_bad[j].support for j in range(len(maybe_do_good))])[::-1]
        for j in range(max(0,min(5-len(do_idx),len(maybe_do_idx)))):
            print()
            print(maybe_do_good[maybe_do_idx[j]].items)
            print("support in good subset selection:")
            print(maybe_do_good[maybe_do_idx[j]].support)
            print("support in bad subset selection:")
            print(maybe_do_bad[maybe_do_idx[j]].support)
        print()
        print("do not:")
        print()
        do_not = []
        maybe_do_not_bad = []
        maybe_do_not_good = []
        for j in range(len(p_union_s_bad_results[idx])):
            exist_equal = False
            for k in range(len(p_union_s_good_results[idx])):
                if p_union_s_bad_results[idx][j].items == p_union_s_good_results[idx][k].items:
                    exist_equal = True
                    if p_union_s_bad_results[idx][j].support > p_union_s_good_results[idx][k].support:
                        maybe_do_not_bad.append(p_union_s_bad_results[idx][j])
                        maybe_do_not_good.append(p_union_s_good_results[idx][k])
            if not exist_equal and not any('acc:' in string for string in p_union_s_bad_results[idx][j].items) and not any('time:' in string for string in p_union_s_bad_results[idx][j].items):
                do_not.append(p_union_s_bad_results[idx][j])
        do_not_idx = np.argsort([j.support for j in do_not])[::-1]
        for j in range(min(5,len(do_not_idx))):
            print()
            print(do_not[do_not_idx[j]].items)
            print("support in good subset selection:")
            print("< " + str(min_support))
            print("support in bad subset selection:")
            print(do_not[do_not_idx[j]].support)
        maybe_do_not_idx = np.argsort([maybe_do_not_bad[j].support /maybe_do_not_good[j].support for j in range(len(maybe_do_not_bad))])[::-1]
        for j in range(max(0,min(5-len(do_not_idx),len(maybe_do_not_idx)))):
            print()
            print(maybe_do_not_good[maybe_do_not_idx[j]].items)
            print("support in good subset selection:")
            print(maybe_do_not_good[maybe_do_not_idx[j]].support)
            print("support in bad subset selection:")
            print(maybe_do_not_bad[maybe_do_not_idx[j]].support)
            

#define the search space.
activation_fun = ["softmax"]
activation_fun_conv = ["elu","relu","tanh","sigmoid","selu"]

filters = OrdinalSpace([10, 600], 'filters') * 14
kernel_size = OrdinalSpace([1, 16], 'k') * 14#CHRIS tweaked
strides = OrdinalSpace([1, 10], 's') * 7#CHRIS tweaked
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

drop_out = ContinuousSpace([1e-5, .9], 'dropout') * 10        # drop_out rate
lr_rate = ContinuousSpace([1e-4, 1.0e-2], 'lr')        # learning rate#CHRIS tweaked
l2_regularizer = ContinuousSpace([1e-5, 1e-2], 'l2')# l2_regularizer


search_space =  stack_sizes * strides * filters *  kernel_size * activation * activation_dense * drop_out * lr_rate * l2_regularizer * step * global_pooling * skstart * skstep * max_pooling * dense_size

#print("searchspace",search_space.levels)

time_model = RandomForest(levels=search_space.levels,n_estimators=1000,workaround=True)
loss_model = RandomForest(levels=search_space.levels,n_estimators=1000,workaround=True)
X = np.atleast_2d([s.tolist() for s in solutions])
time_fitness = np.array([s.time for s in solutions])
loss_fitness = np.array([s.loss for s in solutions])

if do_feature_imp:
    loss_model.fit(X, loss_fitness)
    importances = loss_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in loss_model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    print(len(X[0]))
    print(X.shape)
    print(len(name_array[0]))
    print(len(importances))
    #print(importances)

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, name_array[0][indices[f]], importances[indices[f]]))

    ## Plot the feature importances of the forest
    #plt.figure(figsize=(20,10))
    #plt.title("Feature importances")
    #plt.bar(range(X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
    #plt.xticks(range(X.shape[1]), indices,rotation=60)
    #plt.xlim([-1, X.shape[1]])
    #plt.show()

if do_sens_analysis:
    pass
