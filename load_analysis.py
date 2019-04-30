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
import numbers
from pandas.plotting import parallel_coordinates
import sklearn
from sklearn.cluster import KMeans, DBSCAN
import sklearn.metrics as sm
from apyori import apriori
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

disfunc_time = 80000 #200000 #CHRIS penalty value given to a disfuncitonal network. This differs per experiment
cut = True #needed for derived data such as depth, in case of the cut method, the network is less deep
max_stack = 7 #the number of stacks in the construction method
CIFAR10 = True #do we use CIFAR-10 or MNIST?
img_dim = 32 #CIFAR-10 has 32x32 images

do_spline_fit = False
do_parallel_plot = False
do_pairgrid = False
do_correlations = False
do_k_means = False
do_dbscan = False
do_rule_finding = True
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
vla = {'filters_4': 96, 'stack_6': 4, 'skstep_4': 5, 'k_2': 5, 'stack_5': 4, 'filters_13': 264, 'dropout_4': 0.7522812347648945, 'k_0': 11, 'skstep_2': 6, 'skstart_4': 7, 'filters_11': 190, 's_6': 6, 'filters_10': 171, 'filters_9': 164, 'dropout_5': 0.5061046505734745, 'filters_5': 266, 'dropout_2': 0.17276412402942404, 'k_11': 2, 'k_1': 9, 'stack_3': 0, 'dropout_6': 0.8251804103904812, 'dense_size_1': 3254, 'filters_12': 349, 'l2': 0.0019845517494426227, 'k_8': 8, 'lr': 0.006308299610988853, 's_5': 6, 'k_6': 7, 's_0': 7, 'epoch_sp': 24, 'dropout_8': 0.4334407345524673, 'k_10': 3, 'k_12': 15, 'skstep_1': 3, 's_3': 2, 'global_pooling': False, 'filters_6': 551, 'stack_4': 5, 'skstart_2': 0, 'dropout_3': 0.04315591833160787, 'dense_size_0': 2138, 'max_pooling': True, 'step': True, 'skstart_1': 0, 's_2': 3, 'k_7': 15, 's_4': 8, 'stack_2': 0, 'k_5': 12, 'k_3': 10, 'stack_0': 0, 'dropout_0': 0.006585730795479166, 'filters_8': 258, 'filters_1': 92, 'k_13': 13, 'filters_3': 437, 'stack_1': 7, 's_1': 7, 'k_9': 15, 'dropout_7': 0.722057620926275, 'activ_dense': 'softmax', 'skstart_0': 3, 'filters_2': 223, 'dropout_9': 0.33853188036625337, 'activation': 'tanh', 'skstart_3': 4, 'k_4': 11, 'dropout_1': 0.14516130505677416, 'batch_size_sp': 161, 'filters_7': 239, 'filters_0': 61, 'skstep_3': 9, 'skstep_0': 7}
solutions = [Solution(x=[vla[j] for j in name_array[0]],fitness=1,n_eval=1,index=index_array[0],var_name=name_array[0],loss=1,time=1)]
#TODO end test code removal

print("len(solutions): " + str(len(solutions)))
#print([i.to_dict() for i in solutions])
#y = [np.exp(-i.loss) for i in solutions]#[i.time for i in solutions]
acc_pivot = 0.7
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
        data_lib["elu"] = elu
        data_lib["relu"] = relu
        data_lib["tanh"] = tanh
        data_lib["sigmoid"] = sigmoid
        data_lib["selu"] = selu
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
        data_lib_good["elu"] = elu
        data_lib_good["relu"] = relu
        data_lib_good["tanh"] = tanh
        data_lib_good["sigmoid"] = sigmoid
        data_lib_good["selu"] = selu
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
        data_lib_bad["elu"] = elu
        data_lib_bad["relu"] = relu
        data_lib_bad["tanh"] = tanh
        data_lib_bad["sigmoid"] = sigmoid
        data_lib_bad["selu"] = selu
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

for i in range(max_stack):
    for j in range(len(depth)):
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
    for j in range(len(depth_good)):
        if (not cut) or img_size_good[j] > 1:
            depth_good[j] += data_lib_good["stack_"+str(i)][j]
            num_features_good[j] += data_lib_good["stack_"+str(i)][j] * data_lib_good["filters_" + str(2*i)][j]
            if data_lib_good["stack_"+str(i)][j] > 0:
                avg_dropout_good[j] += data_lib_good["dropout_" + str(i+1)][j]
                dropout_norm_good[j] += 1
                img_size_good[j] = int(np.ceil(img_size_good[j] / data_lib_good["s_" + str(i)][j]))
                if not data_lib_good["max_pooling"][j]:
                    num_features_good[j] += data_lib_good["filters_" + str(2*i+1)][j]
                    depth_good[j] += 1
    for j in range(len(depth_bad)):
        if (not cut) or img_size_bad[j] > 1:
            depth_bad[j] += data_lib_bad["stack_"+str(i)][j]
            num_features_bad[j] += data_lib_bad["stack_"+str(i)][j] * data_lib_bad["filters_" + str(2*i)][j]
            if data_lib_bad["stack_"+str(i)][j] > 0:
                avg_dropout_bad[j] += data_lib_bad["dropout_" + str(i+1)][j]
                dropout_norm_bad[j] += 1
                img_size_bad[j] = int(np.ceil(img_size_bad[j] / data_lib_bad["s_" + str(i)][j]))
                if not data_lib_bad["max_pooling"][j]:
                    num_features_bad[j] += data_lib_bad["filters_" + str(2*i+1)][j]
                    depth_bad[j] += 1
for j in range(len(depth)):
    if data_lib["dense_size_0"][j] > 0:
        avg_dropout[j] += data_lib["dropout_" + str(max_stack+1)][j]
        dropout_norm[j] += 1
    if data_lib["dense_size_1"][j] > 0:
        avg_dropout[j] += data_lib["dropout_" + str(max_stack+2)][j]
        dropout_norm[j] += 1
for j in range(len(depth_good)):
    if data_lib_good["dense_size_0"][j] > 0:
        avg_dropout_good[j] += data_lib_good["dropout_" + str(max_stack+1)][j]
        dropout_norm_good[j] += 1
    if data_lib_good["dense_size_1"][j] > 0:
        avg_dropout_good[j] += data_lib_good["dropout_" + str(max_stack+2)][j]
        dropout_norm_good[j] += 1
for j in range(len(depth_bad)):
    if data_lib_bad["dense_size_0"][j] > 0:
        avg_dropout_bad[j] += data_lib_bad["dropout_" + str(max_stack+1)][j]
        dropout_norm_bad[j] += 1
    if data_lib_bad["dense_size_1"][j] > 0:
        avg_dropout_bad[j] += data_lib_bad["dropout_" + str(max_stack+2)][j]
        dropout_norm_bad[j] += 1

if CIFAR10:
    avg_dropout = [avg_dropout[j]/dropout_norm[j] for j in range(len(avg_dropout))]
    avg_dropout_good = [avg_dropout_good[j]/dropout_norm_good[j] for j in range(len(avg_dropout_good))]
    avg_dropout_bad = [avg_dropout_bad[j]/dropout_norm_bad[j] for j in range(len(avg_dropout_bad))]
else:
    print("ERROR! first implement MNIST dropout normalisation")

img_size = np.array([img_dim] * len(data_lib["stack_0"]))
overlap = np.array([[0]*len(data_lib["stack_0"])] * 5)

print("aaaaaaaaaaaaa")
print(img_size)
print()

for i in range(len(data_lib["stack_0"])):
    current_level = 1
    for j in range(max_stack):
        local_skip_start = [data_lib["skstart_"+str(l)][i] for l in range(5)]
        print("range:")
        print(data_lib["stack_"+str(j)][i])
        for k in range(data_lib["stack_"+str(j)][i]):
            test = [(current_level, local_skip_start[m], current_level - local_skip_start[m] , data_lib["skstep_"+str(m)][i]) for m in range(len(local_skip_start))]
            idx = sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 for m in range(len(local_skip_start))])
            idx -= sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and data_lib["skstep_"+str(n)][i] > 1 and current_level > local_skip_start[n] and m != n and data_lib["skstep_"+str(m)][i] == data_lib["skstep_"+str(n)][i] and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 and (current_level - local_skip_start[n]) % data_lib["skstep_"+str(n)][i] == 0 for m in range(len(local_skip_start)) for n in range(len(local_skip_start))])//2
            if idx > 0:
                overlap[idx-1][i] += 1
            print("current_level: " + str(current_level))
            print("test: " +str(test))
            print("idx: " +str(idx))
            print("img_size:" + str(img_size[i]))
            print("overlap:")
            print(overlap)
            print()
            current_level += 1
        if not data_lib["max_pooling"][i] and data_lib["stack_"+str(j)][i] > 0:
            test = [(current_level, local_skip_start[m], current_level - local_skip_start[m] , data_lib["skstep_"+str(m)][i]) for m in range(len(local_skip_start))]
            idx = sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 for m in range(len(local_skip_start))])
            idx -= sum([current_level > local_skip_start[m] and data_lib["skstep_"+str(m)][i] > 1 and data_lib["skstep_"+str(n)][i] > 1 and current_level > local_skip_start[n] and m != n and data_lib["skstep_"+str(m)][i] == data_lib["skstep_"+str(n)][i] and (current_level - local_skip_start[m]) % data_lib["skstep_"+str(m)][i] == 0 and (current_level - local_skip_start[n]) % data_lib["skstep_"+str(n)][i] == 0 for m in range(len(local_skip_start)) for n in range(len(local_skip_start))])//2
            if idx > 0:
                overlap[idx-1][i] += 1
            print("current_level: " + str(current_level))
            print("test: " +str(test))
            print("idx: " +str(idx))
            print("img_size:" + str(img_size[i]))
            print("overlap:")
            print(overlap)
            print()
            current_level += 1
        print("stride: " + str(data_lib["s_" + str(j)][i]))
        if data_lib["stack_"+str(j)][i] > 0:
            img_size[i] = int(np.ceil(img_size[i] / data_lib["s_" + str(j)][i]))
        if cut and img_size[i] <= 1:
            break
        
print("overlap:")
print(overlap)
print("depth")
print(depth)
print("num_features")
print(num_features)
print("avg_dropout")
print(avg_dropout)
exit(0)
overlap_good = []
overlap_bad = []
for i in range(5):
    data_lib["overlap_"+str(i+1)]=overlap[i]
    data_lib_good["overlap_"+str(i+1)]=overlap_good[i]
    data_lib_bad["overlap_"+str(i+1)]=overlap_bad[i]


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

select = [x for x in data_panda.columns if x != "time" and x != "acc" and x != "activation" and x != "activ_dense"]
selection = data_panda.loc[:, select]
normalizer = selection.max()-selection.min()
for i in range(normalizer.shape[0]):
    if normalizer[i] == 0:
        if data_panda.max()[i] == 0:
            normalizer[i] = 1.0
        else:
            normalizer[i] = data_panda.max()

normalized_df=(selection-selection.min())/normalizer


if do_parallel_plot:
    fig = matplotlib.pyplot.gcf()
    for i in range(normalized_df.shape[0]):
        parallel_coordinates(pd.DataFrame(data=normalized_df.loc[i:i]),'acc',alpha=normalized_df.loc[i:i]['acc'].values)#, colormap=plt.get_cmap("Set2"))
    fig.set_size_inches(180, 105)
    fig.savefig('parallel_coord_plot_cut.png',dpi=100)

if do_pairgrid:
    #g = sns.PairGrid(data_panda_good,vars=['acc','time','stack_0','stack_1','stack_2','stack_3','stack_4','stack_5','stack_6','s_0','s_1','s_2','s_3','s_4','s_5','s_6','filters_0','filters_1','filters_2','filters_3','filters_4','filters_5','filters_6','filters_7','filters_8','filters_9','filters_10','filters_11','filters_12','filters_13','k_0','k_1','k_2','k_3','k_4','k_5','k_6','k_7','k_8','k_9','k_10','k_11','k_12','k_13','dropout_0','dropout_1','dropout_2','dropout_3','dropout_4','dropout_5','dropout_6','dropout_7','dropout_8','dropout_9','lr','l2','global_pooling','skstart_0','skstart_1','skstart_2','skstart_3','skstart_4','skstep_0','skstep_1','skstep_2','skstep_3','skstep_4','dense_size_0','dense_size_1','epoch_sp','batch_size_sp'],hue='acc',palette='GnBu_d')
    #g = sns.PairGrid(data_panda,vars=['acc','time','lr','l2','s_0','s_1','filters_0','filters_1','k_0','k_1','dropout_0','dropout_1','dense_size_0','dense_size_1'],hue='acc',palette='GnBu_d')
    #g = sns.PairGrid(data_panda,vars=['filters_0','filters_1','filters_2','filters_3','filters_4','filters_5','filters_6','filters_7','filters_8','filters_9','filters_10','filters_11','filters_12','filters_13'],hue='acc',palette='GnBu_d')
    #g = sns.PairGrid(data_panda,vars=['k_0','k_1','k_2','k_3','k_4','k_5','k_6','k_7','k_8','k_9','k_10','k_11','k_12','k_13'],hue='acc',palette='GnBu_d')
    #g = sns.PairGrid(data_panda,vars=["softmax","elu","relu","tanh","sigmoid","selu"],hue='acc',palette='GnBu_d')
    g = sns.PairGrid(data_panda,vars=["acc","time","epoch_sp","batch_size_sp","depth"],hue='acc',palette='GnBu_d')
    g.map(plt.scatter)
    #g.map_upper(plt.scatter)
    #g.map_lower(sns.kdeplot)
    #g.map_diag(sns.kdeplot, lw=3, legend=False);

    #plt.savefig('load_analysis_output.png')
    plt.savefig('load_analysis_output_cut_epoch_batch_depth.png')

if do_correlations:
    corr_pivot = 0.3
    #parallel_coordinates(data_panda, class_column='acc', cols=['lr','l2'])
    select = [x for x in data_panda.columns if x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
    #select = [x for x in data_panda.columns if x == "acc" or x == "time" or x == "depth" or x == "epoch_sp" or x == "batch_size_sp"]
    selection = data_panda.loc[:, select]
    correlations = selection.corr()
    
    print()
    print('correlations greater than '+ str(corr_pivot) + ' on all data')
    print()
    corr_pivot_labels = set([])
    for i in correlations:
        for j in correlations:
            if correlations.loc[i,j] != 1.0 and abs(correlations.loc[i,j]) >= corr_pivot:
                print(i,j)
                print(correlations.loc[i,j])
                corr_pivot_labels = corr_pivot_labels.union(set([i,j]))
    corr_pivot_labels = list(corr_pivot_labels)
    corr_select = selection[corr_pivot_labels].corr()
    #plt.matshow(correlations)
    sns.heatmap(corr_select,xticklabels=corr_select.columns.values,yticklabels=corr_select.columns.values,cmap='coolwarm')
    #correlations.style.background_gradient(cmap='coolwarm').set_precision(2)
    plt.show()
    
    select_good = [x for x in data_panda_good.columns if x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
    selection_good = data_panda_good.loc[:, select_good]
    correlations_good = selection_good.corr()
    
    print()
    print('correlations greater than '+ str(corr_pivot) + ' on good data')
    print()
    corr_pivot_labels_good = set([])
    for i in correlations_good:
        for j in correlations_good:
            if correlations_good.loc[i,j] != 1.0 and abs(correlations_good.loc[i,j]) >= corr_pivot:
                print(i,j)
                print(correlations_good.loc[i,j])
                corr_pivot_labels_good = corr_pivot_labels_good.union(set([i,j]))
    corr_pivot_labels_good = list(corr_pivot_labels_good)
    corr_select_good = selection_good[corr_pivot_labels_good].corr()
    sns.heatmap(corr_select_good,xticklabels=corr_select_good.columns.values,yticklabels=corr_select_good.columns.values,cmap='coolwarm')
    plt.show()

    select_bad = [x for x in data_panda_bad.columns if x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
    selection_bad = data_panda_bad.loc[:, select_bad]
    correlations_bad = selection_bad.corr()
    
    print()
    print('correlations greater than '+ str(corr_pivot) + ' on bad data')
    print()
    corr_pivot_labels_bad = set([])
    for i in correlations_bad:
        for j in correlations_bad:
            if correlations_bad.loc[i,j] != 1.0 and abs(correlations_bad.loc[i,j]) >= corr_pivot:
                print(i,j)
                print(correlations_bad.loc[i,j])
                corr_pivot_labels_bad = corr_pivot_labels_bad.union(set([i,j]))
    corr_pivot_labels_bad = list(corr_pivot_labels_bad)
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
    
    select = [x for x in data_panda.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
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
                for d in selection[x]:
                    data_discrete.append([string_cast[min(n_sections-1,int((float(d) - min_val)/(max_val-min_val)*n_sections))]])
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
    
    select_good = [x for x in data_panda_good.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
    selection_good = data_panda_good.loc[:, select_good]
    
    data_discrete_good = []
    first = True
    for x in selection_good.columns:
        if x != "activation" and x != "activ_dense":
            if first:
                for d in selection_good[x]:
                    data_discrete_good.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))]])
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
    
    select_bad = [x for x in data_panda_bad.columns if x != "time" and x != "acc" and x != "elu" and x != "relu" and x != "tanh" and x != "sigmoid" and x != "selu"]
    selection_bad = data_panda_bad.loc[:, select_bad]

    data_discrete_bad = []
    first = True
    for x in selection_bad.columns:
        if x != "activation" and x != "activ_dense":
            if first:
                for d in selection_bad[x]:
                    data_discrete_bad.append([string_cast_dict[x][min(n_sections-1,int((float(d) - min_val_dict[x])/(max_val_dict[x]-min_val_dict[x])*n_sections))]])
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
        print(patterns[idx_sort[i]].items)
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
        print(good_patterns[idx_sort[i]].items)
        print(good_patterns[idx_sort[i]].support)

    bad_patterns = []
    print('\nbad, 1 items or more top 10 support:\n')
    for i in results_bad:
        if len(i.items) >= 1:
            bad_patterns.append(i)
    idx_sort = np.argsort([i.support for i in bad_patterns])[::-1]
    for i in range(min(len(idx_sort),10)):
        print(bad_patterns[idx_sort[i]].items)
        print(bad_patterns[idx_sort[i]].support)
    
    good_patterns_not_in_bad = []
    print('\ngood patterns not in bad (1 items or more) top 10:\n')
    for i in results_good:
        exist = False
        for j in results_bad:
            if i.items == j.items:
                exist = True
                break
        if not exist:
            good_patterns_not_in_bad.append(i)
    idx_sort = np.argsort([i.support for i in good_patterns_not_in_bad])[::-1]
    for i in range(min(len(idx_sort),10)):
        print(good_patterns_not_in_bad[idx_sort[i]].items)
        print(good_patterns_not_in_bad[idx_sort[i]].support)
    
    bad_patterns_not_in_good = []
    print('\nbad patterns not in good (1 items or more) top 10:\n')
    for i in results_bad:
        exist = False
        for j in results_good:
            if i.items == j.items:
                exist = True
                break
        if not exist:
            bad_patterns_not_in_good.append(i)
    idx_sort = np.argsort([i.support for i in bad_patterns_not_in_good])[::-1]
    for i in range(min(len(idx_sort),10)):
        print(bad_patterns_not_in_good[idx_sort[i]].items)
        print(bad_patterns_not_in_good[idx_sort[i]].support)
    
    union_patterns = []
    bad_union_support = []
    print('\nunion, 1 items or more:\n')
    for i in results_good:
        if len(i.items) >= 1:
            for j in results_bad:
                if len(j.items) >= 1 and i.items == j.items:
                    union_patterns.append(i)
                    bad_union_support.append(j.support)
                    print(i.items)
                    print(i.support)
                    print(j.support)
    
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
    for i in range(min(len(top_good_idx),10)):
        idx = top_good_idx[i]
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
        for j in range(len(p_union_s_good_results[idx])):
            do = []
            exist_equal = False
            for k in range(len(p_union_s_bad_results[idx])):
                if p_union_s_good_results[idx][j].items == p_union_s_bad_results[idx][k].items:
                    exist_equal = True
                    break
            if not exist_equal and not any('acc:' in string for string in p_union_s_good_results[idx][j].items) and not any('time:' in string for string in p_union_s_good_results[idx][j].items):
                do.append(p_union_s_good_results[idx][j])
        do_idx = np.argsort([j.support for j in do])[::-1]
        for j in range(min(5,len(do_idx))):
            print(do[do_idx[j]].items)
            print("support in good subset selection:")
            print(do[do_idx[j]].support)
        print()
        print("do not:")
        print()
        for j in range(len(p_union_s_bad_results[idx])):
            do_not = []
            exist_equal = False
            for k in range(len(p_union_s_good_results[idx])):
                if p_union_s_bad_results[idx][j].items == p_union_s_good_results[idx][k].items:
                    exist_equal = True
                    break
            if not exist_equal and not any('acc:' in string for string in p_union_s_bad_results[idx][j].items) and not any('time:' in string for string in p_union_s_bad_results[idx][j].items):
                do_not.append(p_union_s_bad_results[idx][j])
        do_not_idx = np.argsort([j.support for j in do_not])[::-1]
        for j in range(min(5,len(do_not_idx))):
            print(do_not[do_not_idx[j]].items)
            print("support in good subset selection:")
            print(do_not[do_not_idx[j]].support)
            

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

time_model = RandomForest(levels=search_space.levels,n_estimators=10,workaround=True)
loss_model = RandomForest(levels=search_space.levels,n_estimators=10,workaround=True)
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
