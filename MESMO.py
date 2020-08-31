# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:34:01 2018

@author: Syrine Belakaria
"""
import os
import numpy as np
from GPmodel import GaussianProcess
from singlemes import MaxvalueEntropySearch
from scipy.optimize import minimize as scipyminimize
from platypus import NSGAII, Problem, Real
import sobol_seq
from pygmo import hypervolume


######################Algorithm input##############################
paths='.'


from benchmark_functions import branin,Currin
functions=[branin,Currin]
d=2
referencePoint = [1e5]*len(functions)

total_iterations=100
seed=0
np.random.seed(seed)
intial_number=1
bound=[0,1]
sample_number=1

Fun_bounds = [bound]*d
grid = sobol_seq.i4_sobol_generate(d,1000,np.random.randint(0,100))
design_index = np.random.randint(0, grid.shape[0])

###################GP Initialisation##########################

GPs=[]
Multiplemes=[]
for i in range(len(functions)):
    GPs.append(GaussianProcess(d))
for k in range(intial_number):
    exist=True
    while exist:
        design_index = np.random.randint(0, grid.shape[0])
        x_rand=list(grid[design_index : (design_index + 1), :][0])
        if (any((x_rand == x).all() for x in GPs[0].xValues))==False:
            exist=False
    for i in range(len(functions)):
        GPs[i].addSample(np.asarray(x_rand),functions[i](x_rand,d))
        
for i in range(len(functions)):   
    GPs[i].fitModel()
    Multiplemes.append(MaxvalueEntropySearch(GPs[i]))
    

#### write the initial points into file
input_output= open(os.path.join(paths,'input_output.txt'), "a")
for j in range(len(GPs[0].yValues)):    
    input_output.write(str(GPs[0].xValues[j])+'---'+str([GPs[i].yValues[j] for i in range(len(functions))]) +'\n' )
input_output.close()

##################### main loop ##########

for l in range(total_iterations):
        
    for i in range(len(functions)):
        Multiplemes[i]=MaxvalueEntropySearch(GPs[i])
        Multiplemes[i].Sampling_RFM()
    max_samples=[]
    for j in range(sample_number):
        for i in range(len(functions)):
            Multiplemes[i].weigh_sampling()
        cheap_pareto_front=[]
        def CMO(xi):
            xi=np.asarray(xi)
            y=[Multiplemes[i].f_regression(xi)[0][0] for i in range(len(GPs))]
            return y
        
        problem = Problem(d, len(functions))
        problem.types[:] = Real(bound[0], bound[1])
        problem.function = CMO
        algorithm = NSGAII(problem)
        algorithm.run(1500)
        cheap_pareto_front=[list(solution.objectives) for solution in algorithm.result]
#########picking the max over the pareto: best case
        maxoffunctions=[-1*min(f) for f in list(zip(*cheap_pareto_front))]
        max_samples.append(maxoffunctions)

    def mesmo_acq(x):
        multi_obj_acq_total=0
        for j in range(sample_number):
            multi_obj_acq_sample=0
            for i in range(len(functions)):
                multi_obj_acq_sample=multi_obj_acq_sample+Multiplemes[i].single_acq(x,max_samples[j][i])
            multi_obj_acq_total=multi_obj_acq_total+multi_obj_acq_sample
        return (multi_obj_acq_total/sample_number)

    
    # l-bfgs-b acquisation optimization
    x_tries = np.random.uniform(bound[0], bound[1],size=(1000, d))
    y_tries=[mesmo_acq(x) for x in x_tries]
    sorted_indecies=np.argsort(y_tries)
    i=0
    x_best=x_tries[sorted_indecies[i]]
    while (any((x_best == x).all() for x in GPs[0].xValues)):
        i=i+1
        x_best=x_tries[sorted_indecies[i]]
    y_best=y_tries[sorted_indecies[i]]
    x_seed=list(np.random.uniform(low=bound[0], high=bound[1], size=(1000,d)))    
    for x_try in x_seed:
        result = scipyminimize(mesmo_acq, x0=np.asarray(x_try).reshape(1, -1), method='L-BFGS-B', bounds=Fun_bounds)
        if not result.success:
            continue
        if ((result.fun<=y_best) and (not (result.x in np.asarray(GPs[0].xValues)))):
            x_best=result.x
            y_best=result.fun



#---------------Updating and fitting the GPs-----------------   
    for i in range(len(functions)):
        GPs[i].addSample(x_best,functions[i](list(x_best),d))
        GPs[i].fitModel()

    ############################ write Input output into file ##################
    input_output= open(os.path.join(paths,'input_output.txt'), "a")    
    input_output.write(str(GPs[0].xValues[-1])+'---'+str([GPs[i].yValues[-1] for i in range(len(functions))]) +'\n' )
    input_output.close()
    
    ############################ write hypervolume into file##################
                
#    current_hypervolume= open(os.path.join(paths,'hypervolumes.txt'), "a")    
#    simple_pareto_front_evaluations=list(zip(*[GPs[i].yValues for i in range(len(functions))]))
#    print("hypervolume ", hypervolume(-1*(np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
#    current_hypervolume.write('%f \n' % hypervolume(-1*(np.asarray(simple_pareto_front_evaluations))).compute(referencePoint))
#    current_hypervolume.close()
