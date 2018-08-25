import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import copy

def s_metric(expected, solutions,n_left,ref_time=None,ref_loss=None):
    par = pareto(solutions)#CHRIS pareto front of existing solutions
    sol_and_exp = copy.deepcopy(solutions)
    sol_and_exp.append(expected)#CHRIS existing solutions and expected solution unified
    par_and_exp = pareto(sol_and_exp)#CHRIS pareto front of existing solutions and expected solution unified
    
    #CHRIS calculate epsilon vector
    if len(par) >= 1:
        max_time = max([i.time for i in par])
        min_time = min([i.time for i in par])
        max_loss = max([i.loss for i in par])
        min_loss = min([i.loss for i in par])
        d_lamb = np.array([max_time-min_time, max_loss-min_loss])
        m = 2 #CHRIS because we have two objectives
        c = 1-(1/(2**m))#TODO_CHRIS this constant might not be ideal for this specific problem
        eps = (d_lamb/len(par))*c*n_left
    else:
        eps = np.array([float('inf'),float('inf')])
    exp_min_eps = copy.deepcopy(expected)#CHRIS easier to shift expected point rather than entire paretofront
    exp_min_eps.time -= eps[0]
    exp_min_eps.loss -= eps[1]
    val=0.0

    skip_hyp_vol = False #CHRIS in case that the expected point lies outside the rectangle formed by the reference point and (0,0), and that the expected point is also not dominated (this can happen), the hyper volume must not be calculated
    if (not ref_time is None) and (expected.time > ref_time):
        skip_hyp_vol = True
    if (not ref_loss is None) and (expected.loss > ref_loss):
        skip_hyp_vol = True

    if (not dominated(exp_min_eps, par)) and (not skip_hyp_vol):
        #CHRIS non epsilon dominated solutions receive benefit of hypervolume
        val += (hyper_vol(par_and_exp,sol_and_exp,ref_time=ref_time,ref_loss=ref_loss)-hyper_vol(par,sol_and_exp,ref_time=ref_time,ref_loss=ref_loss))
    else:
        #CHRIS epsilon dominated solutions only receive penalty for inferior objectives
        #val -= penalty(exp_min_eps, par)#CHRIS changed exp_min_eps to expected because pareto front values got too much penalty. With exp_min_eps is the hump version, With epxpected is the normal version
        val -= eps_penalty(expected,par)
    if dominated(expected, par):
        #CHRIS dominated expected point receives penalty
        val -= penalty(expected,solutions)
    #print('pareto-front:')
    #if len(par) > 0:
    #    for x in par:
    #        print('time: ' + str(x.time) + ' loss: ' + str(x.loss) + ' fit:' + str(x.fitness))
    return val

def penalty(expected,solutions):
    val = 0.0
    for i in range(len(solutions)):
        if dominated(expected, [solutions[i]]):
            pen = (1+expected.loss - solutions[i].loss)*(1+expected.time-solutions[i].time)-1
            val += pen
    return val

def sort_par(par):
    #sort pareto front ascending on loss
    for i in range(len(par)):
        for j in range(i+1, len(par)):
            if (par[i].loss < par[j].loss):
                help = par[i]
                par[i] = par[j]
                par[j] = help
    return par

def hyper_vol(par, solutions,ref_time=None,ref_loss=None):
    #set maximum values in pareto front as reference points
    if len(solutions) == 0:
        return 0.0
    if len(par) == 0:
        return 0.0
    if ref_time == None or ref_loss == None:
        ref_time = solutions[0].time
        ref_loss = solutions[0].loss
        for i in range(1,len(solutions)):
            if (solutions[i].time > ref_time):
                ref_time = solutions[i].time
            if (solutions[i].loss > ref_loss):
                ref_loss = solutions[i].loss
        ref_time += 1.0
        ref_loss += 1.0
    
    loc_par = copy.deepcopy(par)

    #remove entries in pareto front bigger than the reference point (can only happen if ref_time and ref_loss are given as parameter to function)
    i = 0
    while i < len(loc_par):
        if loc_par[i].time > ref_time or loc_par[i].loss > ref_loss:
            del loc_par[i]
        else:
            i += 1
    #sort pareto front
    loc_par = sort_par(loc_par)

    #calculate hypervolume
    if len(loc_par) > 0:
        vol = (ref_loss - loc_par[0].loss) * (ref_time - loc_par[0].time)
        for i in range(1,len(loc_par)):
            vol += (loc_par[i-1].loss - loc_par[i].loss) * (ref_time - loc_par[i].time)
    else:
        vol = 0.0
    return vol

def dominated(point, solutions):
    #returns whether <point> is dominated by the set <solutions>
    for i in range(len(solutions)):
        if point.loss < solutions[i].loss:
            continue
        if point.time < solutions[i].time:
            continue
        if (point.loss == solutions[i].loss) and (point.time == solutions[i].time):
            continue
        return True
    return False

def pareto(solutions):
    #returns the set of non-dominated solutions in <solutions>
    par = []
    for i in range(len(solutions)):
        if not dominated(solutions[i], solutions):
            par.append(copy.deepcopy(solutions[i]))
    return par

def eps_penalty(solution,par):
    #TODO_CHRIS penalty of epsilon point by dominated paretofron only, reuse penalty function
    val = 0.0
    for i in range(len(par)):
        if solution.time > par[i].time:
            prod1 = 1.0 + solution.time - par[i].time
        else:
            prod1 = 1.0
        if solution.loss > par[i].loss:
            prod2 = 1.0 + solution.loss - par[i].loss
        else:
            prod2 = 1.0
        val += -1.0 + prod1 * prod2
    return val

#def test_func_1(x):
#    return x+np.random.rand()
#
#def test_func_2(x):
#    return x**2 + np.random.rand()
#
##TODO_CHRIS Point just for test code
#class Point:
#    def __init__(self,x):
#        self.x = x
#        self.time = test_func_1(self.x)+80
#        self.loss = test_func_2(self.x)+80
#        self.fitness = 0.0
#
#def test(n_left):
#    n = 100
#    solutions = [0.0] * n
#    for i in range(n):
#        solutions[i] = Point(1-i/n)
#    #solutions[0].time = 0.0
#    #solutions[0].loss = 0.0
#
#
#    for i in range(len(solutions)):
#        other = copy.deepcopy(solutions)
#        del other[i]
#        solutions[i].fitness = s_metric(solutions[i],other,n_left)
#
#    tests = [0.0] * 10
#    for i in range(len(tests)):
#        tests[i] = Point(1-i/n)
#        tests[i].time -= 80
#        tests[i].loss -= 80
#        tests[i].fitness = s_metric(tests[i],solutions,n_left)
#
#    x = [0.0]*n
#    y = [0.0]*n
#    z = [0.0]*n
#
#    for i in range(n):
#        x[i] = solutions[i].time
#        y[i] = solutions[i].loss
#        z[i] = solutions[i].fitness
#
#    a = [0.0] * len(tests)
#    b = [0.0] * len(tests)
#    c = [0.0] * len(tests)
#    for i in range(len(tests)):
#        a[i] = tests[i].time
#        b[i] = tests[i].loss
#        c[i] = tests[i].fitness
#    for flip in a:
#        x.append(flip)
#    for flip in b:
#        y.append(flip)
#    for flip in c:
#        z.append(flip)
#
#
#    plt.scatter(x,y,c = z)
#    plt.xlabel('time')
#    plt.ylabel('loss')
#    plt.show()
#
#test()
