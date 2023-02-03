#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas
import gurobipy as gp
from gurobipy import GRB
#import implicit 
import pandas as pd
import numpy as np
#from tqdm.notebook import tqdm
import glob
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import statistics
#from scipy.sparse import csr_matrix, dok_matrix
import _pickle as cPickle
import sys
#import sklearn
#from sklearn import preprocessing
import time


# In[6]:

#def exp(u,i,uid_feature_c_idx_org,pid_feature_c_idx_org,E_npy_org,prob_selling_org,cost_val, intMaxVal):
def gini(arr):
    #arr is an array of size = number_of_reseller and will contain number of orders/items purchaged by reseller on shpshy.
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def NashMax(fileName,instance,L,alpha,epsilon,R2_option):
    v = np.load(f"{fileName}")
    intMaxVal=int(max(np.sum(v,axis=1)))
    users = v.shape[0]
    items = v.shape[1]

    L1 = L - epsilon
    L2 = L + epsilon
    
    exp_type = "NASHMAX"
    R1 = int(alpha*L1*users/items)
    if(R2_option == 2):
        R2 = R1 * 2
    elif(R2_option == 3):
        R2 = R1 + 1
    else :
        R2 = users
    
    print("users->",users,":",L1,"-",L2)
    print("items->",items,":",R1,"-",R2)

    # In[107]:

    begin_f = time.time()
    ## Lets define model now
    x = []
    for i in range(users):
        x.append([])
    for i in range(users):
        for j in range(items):
            x[i].append(None)

    gamma_var = []
    for i in range(users):
        gamma_var.append(None)


    # In[108]:


    m =  gp.Model()
    for i in range(users):
        for j in range(items):
            x[i][j] = m.addVar(vtype= GRB.BINARY, name=f"x_{i}_{j}")
    for i in range(users):
        gamma_var[i] = m.addVar( lb=0.0, ub=GRB.INFINITY,vtype= GRB.CONTINUOUS, name=f"gamma_var_{i}")


    # In[109]:


    objective = gp.LinExpr()
    for i in range(0,users):
            objective.addTerms(1.0,gamma_var[i])
    m.setObjective(objective, GRB.MAXIMIZE)


    # In[110]:


    ## Setting constraints

    ## On user side
    for i in range(0,users):
        constr1 = gp.LinExpr()
        for j in range(0,items):
             constr1.addTerms(1.0,x[i][j])
        m.addConstr(constr1,GRB.GREATER_EQUAL,L1)

    for i in range(0,users):
        constr1 = gp.LinExpr()
        for j in range(0,items):
             constr1.addTerms(1.0,x[i][j])
        m.addConstr(constr1,GRB.LESS_EQUAL,L2)


    # In[111]:


    ## On item side
    for i in range(0,items):
        constr1 = gp.LinExpr()
        for j in range(0,users):
             constr1.addTerms(1.0,x[j][i])
        m.addConstr(constr1,GRB.GREATER_EQUAL,R1)
        #m.addConstr(constr1,GRB.EQUAL,c2)
        
    for i in range(0,items):
        constr1 = gp.LinExpr()
        for j in range(0,users):
             constr1.addTerms(1.0,x[j][i])
        m.addConstr(constr1,GRB.LESS_EQUAL,R2)

    begin = time.time()
    for i in range(users):
        for k in range(1,intMaxVal,2):
            constr1 = gp.LinExpr()
            for j in range(0,items):
                constr1.addTerms(v[i][j]*(np.log(k+1)-np.log(k)),x[i][j])
            constr1.addTerms(-1,gamma_var[i])
            m.addConstr(constr1,GRB.GREATER_EQUAL,-1*np.log(k)+(np.log(k+1)-np.log(k))*k)
    end = time.time()
    time_taken1 = end-begin
    #print("Time taken (constrain) is: ",time_taken1)



    begin = time.time()
    m.optimize()
    end = time.time()
    time_taken2 = end-begin
    time_takenf = end-begin_f
    #print("Time taken (optimize)is: ",time_taken2)
    output = []
    output.append(users)
    output.append(items)
    output.append(L1)
    output.append(L2)
    output.append(R1)
    output.append(R2)
    #output.append(time_taken1)
    #output.append(time_taken2)
    output.append(time_takenf)
    #output.append(intMaxVal)
    output.append(exp_type)
    
    w = np.zeros((users, items))
    for i in range(0,users):
        for j in range(0,items):
            w[i][j] = x[i][j].X
    fname = exp_type+"_instance"+str(instance)+"_L"+str(L)+"_epsilon"+str(epsilon)+"_R1"+str(R1)+"_R2"+str(R2)+"_alpha"+str(alpha)
    
    directory=f"results/{exp_type}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f"{directory}/{fname}.npy",w) 
   
    
    check = True
    print()
    for i in range(users):
        Vi_Ai = np.sum(w[i,:]*v[i,:])
        for j in range(users):
            #print(i,j)
            Vi_Aj = w[j,:]*v[i,:]
            val = np.sum(Vi_Aj)-np.max(Vi_Aj)
            if(Vi_Ai<val):
                print("Not EF1")
                check = False
    
    res_direct = w
    users_alloc = np.zeros(res_direct.shape[0])
    items_alloc = np.zeros(res_direct.shape[1])
    utill_U = np.zeros(res_direct.shape[0])
    utill_P = np.zeros(res_direct.shape[1])
    for i in range(res_direct.shape[0]):
        for j in range(res_direct.shape[1]):
            users_alloc[i] += res_direct[i][j]
            items_alloc[j] += res_direct[i][j]
            utill_U[i] += res_direct[i][j]*v[i][j]
            utill_P[j] += res_direct[i][j]*v[i][j]
#     output = []
#     output.append(instance)
#     output.append(L1)
#     output.append(L2)
#     output.append(R1)
#     output.append(R2)
#     output.append(time_takenf)
#     output.append(sum(np.log(utill_U)))
#     output.append(np.sum(utill_U))
#     output.append(np.max(utill_U)-np.min(utill_U))
#     print(output)
    print("algo: ",exp_type)
    print("instance: ",instance)
    print("L1: ",L1)
    print("L2: ",L1)
    print("R1: ",R1)
    print("R2: ",R2)
    print("log_Nash: ",sum(np.log(utill_U)))
    print("Revenue: ",np.sum(utill_U))
    print("Income Gap: ",np.max(utill_U)-np.min(utill_U))
    print("Time taken: ",time_takenf)
    print("Minimum Utility: ",np.min(utill_U))
    print("Gini re-sellers: ",gini(utill_U))
    print("Gini Products: ",gini(utill_P))
    
    output2 = f"{exp_type},{instance},{L1},{L2},{R1},{R2},{sum(np.log(utill_U))},{np.sum(utill_U)},{np.max(utill_U)-np.min(utill_U)},{time_takenf},{alpha},{np.min(utill_U)},{gini(utill_U)},{gini(utill_P)},\n"
    result_file = f"results/{exp_type}.txt"              
    f= open(result_file,"a+")
    f.write(str(output2))
    f.close()

    
if __name__ == '__main__':
    instance = int(sys.argv[1])
    L = int(sys.argv[2])
    alpha = int(sys.argv[3])
    epsilon = int(sys.argv[4])
    R2_option = int(sys.argv[5])
    fileName = f"../scripts/runtime_results/randomInstance/instance{instance}_v100_u100_randint_1000.npy"
    NashMax(fileName,instance,L,alpha,epsilon,R2_option)
    


