from mipego.mipego import Solution
import random

def quicksort_par(par, lo, hi):
    if lo < hi:
        p = partition_par(par, lo, hi)
        quicksort_par(par, lo, p - 1 )
        quicksort_par(par, p + 1, hi)

def partition_par(par, lo, hi):
    pivot = par[hi].loss
    i = lo
    for j in range(lo, hi):
        if par[j].loss < pivot:
            help = par[i]
            par[i] = par[j]
            par[j] = help
            i = i + 1
    help = par[i]
    par[i] = par[hi]
    par[hi] = help
    return i

n = 100

solutions = [Solution([1]) for i in range(n)]
for i in range(n):
    solutions[i].loss = i

random.shuffle(solutions)
print([x.loss for x in solutions])
quicksort_par(solutions,0,len(solutions)-1)
print([x.loss for x in solutions])
