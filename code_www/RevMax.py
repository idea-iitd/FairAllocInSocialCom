#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import _pickle as cPickle
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


# In[ ]:


from gurobipy import *
import numpy as np
import time
from scipy.sparse import csr_matrix
import collections
import scipy.stats as sp

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


def linkmatr(num_left,num_right):
    #Creates link matrix A used to define degree constraints for all nodes
    num_nodes=num_left+num_right
    str1=[1]*num_right
    str2=[0]*num_right
    A=[None]*(num_nodes)
    for i in range(num_left):
        A[i]=str2*num_left
        #print A[i]
        idx=num_right*i
        A[i][idx:idx+num_right]=str1
    for j in range(num_right):
        A[num_left+j]=str2*num_left
        idx=[j+num_right*l for l in range(num_left)]
        for k in range(num_left):
            A[num_left+j][idx[k]]=1      
    return A


def b_matching(num_left,num_right, W,lda,uda,ldp,udp):
    ##D and W are list
    try:
        # Create a new model
        m = Model("mip1")
        #m.setParam("OutputFlag", 0);
        #m.setParam("MIPFocus", 1)
        

        total_nodes = num_left+num_right
        total_vars = num_left*num_right
        
        if((num_left*lda> num_right*udp) or (num_right*ldp>num_left*uda)):
            print('Infeasible Problem')
            return
        
        #Maximum Number of authors matched to node paper
        Dmax=list(udp*np.ones((total_nodes,)))
        
        #Minimum Number of authors matched to a paper
        Dmin=list(ldp*np.ones((total_nodes,)))
        
        #Minimum Number of papers matched to an author
        Dmina=list(lda*np.ones((total_nodes,)))
        
        #Maximum Number of papers matched to author
        Dmaxa=list(uda*np.ones((total_nodes,)))

        
        A=linkmatr(num_left,num_right)
        x = {}
        for j in range(total_vars):
          x[j] = m.addVar(vtype=GRB.BINARY, name="x"+str(j))

        #Set objective
        m.setObjective((quicksum(W[i]*x[i] for i in range(total_vars))), GRB.MAXIMIZE)
        
        #constraint on paper cardinality
        for i in range(num_left,total_nodes):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmax[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmin[i])
            
        #constraint on authors
        for i in range(num_left):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmaxa[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmina[i])    

        #m.write("lp.mps")    
        # Optimize
        m.optimize()   
        print("ILP finished")
        res=np.zeros((num_left,num_right))
        for i in range(num_left):
            for j in range(num_right):
                idx=num_right*i+j
                res[i,j]=m.getVars()[idx].x

        return res  
    
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')


def RevMax(fileName,instance,L,alpha,epsilon,R2_option):
    v = np.load(f"{fileName}")
    intMaxVal=int(max(np.sum(v,axis=1)))
    users = v.shape[0]
    items = v.shape[1]

    L1 = L - epsilon
    L2 = L + epsilon
    
    exp_type = "REVMAX"
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
    lda, uda, ldp, udp = L1, L2, R1, R2
    W=list(np.ravel(v))       
    #WBM weighed matching
    res_direct=np.round(b_matching(users, items,W ,lda ,uda ,ldp ,udp))
    end = time.time()
    time_takenf = end-begin_f
    #print("Time taken (optimize)is: ",time_takenf)
    
    fname = exp_type+"_instance"+str(instance)+"_L"+str(L)+"_epsilon"+str(epsilon)+"_R1"+str(R1)+"_R2"+str(R2)+"_alpha"+str(alpha)
    
    directory=f"results/{exp_type}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f"{directory}/{fname}.npy",res_direct) 
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
    w=res_direct
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
    RevMax(fileName,instance,L,alpha,epsilon,R2_option)
