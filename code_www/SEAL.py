#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import os
from statistics import mean
#from . import utils
#import utils


def swap_for_cardinality(fin,V,R):
    items = V.shape[1]
    users = V.shape[0]
    P_count = np.zeros(items)
    A = np.zeros((users,items))
    for i in range(0,users):
        for j in fin[i]:
            A[i][j] += 1
                
    for i in range(users):
        for j in range(items):
            P_count[j] += A[i][j]
    #print(P_count)


    P_less = np.zeros(items)
    P_less_list1 = []
    P_less_list2 = []
    for j in range(items):
        if(P_count[j] < R):
            P_less[j] = 1
            P_less_list1.append(j)
            P_less_list2.append(R-P_count[j])
    #print(P_less, P_less_list1, P_less_list2)
    #print(P_less_list1)
    #print(P_less_list2)
    if len(P_less_list1) == 0:
        return fin
    for p in P_less_list1:
        for c in range(0,int(R-P_count[p])):
            print("iterartion",p,c)
            P_eligible = np.where(P_count > R, 1, 0)
            U_eligible = np.where(A[:,p] > 0, 0, 1)
            #print(P_eligible, U_eligible)
            A2 = np.dot(U_eligible.reshape(users,1), P_eligible.reshape(1,items))
            #print(A2)
            A2 = A2*A
            A3 = A2*V
            #print(A3)
            V2 = np.zeros((users,items))
            for j in range(items):
                V2[:,j]=V[:,p].T
            #print(V2)
            A4 = A2 * V2
            #print(A4)
            A5 = A3 - A4
            #print(A5)   
            i_min = -1
            j_min = -1
            min_val = np.max(V)
            for i in range(users):
                for j in range(items):
                    if A2[i][j] > 0 and A5[i][j] < min_val:
                        min_val = A5[i][j]
                        i_min = i
                        j_min = j
            #print(i_min,j_min)            
            A[i_min][j_min] = 0
            A[i_min][p] = 1
            P_count[p] += 1
            P_count[j_min] -= 1
            #print("new allocation:")
            #print(A,P_count)
    fin2 = []
    for i in range(users):
        fin2.append([])
    for i in range(0,users):
        for j in range(0,items):
            if A[i][j] == 1:
                fin2[i].append(j)
    fin2_final = np.array(fin2)
    return fin2_final

def SEAL(file, L, R, R2):
    
    v = np.load(file)
    users = v.shape[0]
    items = v.shape[1]
    count_to_buy = np.zeros(items)
    utility = np.zeros(users)
    for i in range(0,items):
        count_to_buy[i]= R

    try:
       import queue
    except ImportError:
       import Queue as queue 

    A = []
    for i in range(users):
        A.append([])
    start_time = time.time()
    each_user_get = L
    for i in range(0,each_user_get):
        pq1 = queue.PriorityQueue()
        for i in range(0,users):
            pq1.put((utility[i],i))
        while not pq1.empty() :
            (rev,user_idx) = pq1.get()
            prod_indices = np.where(count_to_buy>0)[0]
            if(len(prod_indices)!=0):
                store_feature = np.array([1 if (i in prod_indices and  i not in A[user_idx]) else 0 for i in range(items)])
                store_feature = store_feature * v[user_idx]
                idx = np.argmax(store_feature)
                count_to_buy[idx] -= 1
                A[user_idx].append(idx)
                utility[user_idx] += store_feature[idx]
            else :
                prod_indices = np.where(count_to_buy>R-R2)[0]
                store_feature = np.array([1 if (i in prod_indices and  i not in A[user_idx]) else 0 for i in range(items)])
                #store_feature = np.array([1 if i not in A[user_idx] else 0 for i in range(items)])
                store_feature = store_feature * v[user_idx]
                idx = np.argmax(store_feature)
                count_to_buy[idx] -= 1
                A[user_idx].append(idx)
                utility[user_idx] += store_feature[idx]

    #print("Algorithm Nash_RR_Greedy has terminated.")
    time_taken = (time.time() - start_time)
    fin = np.array(A)
    fin = swap_for_cardinality(fin,v,R)
    print(f"Nash_RR_Greedy Time taken ->{time_taken} seconds")

    #fin = np.array(A)
    return fin


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


if __name__ == '__main__':
    instance = int(sys.argv[1])
    L = int(sys.argv[2])
    alpha = int(sys.argv[3])
    epsilon = int(sys.argv[4])
    epsilon = 0
    R2_option = int(sys.argv[5])
    fileName = f"data/instance{instance}_v100_u100_randint_1000.npy"
    v = np.load(f"{fileName}")
    users = v.shape[0]
    items = v.shape[1]

    L1 = L - epsilon
    L2 = L + epsilon
    
    exp_type = "SEAL"
    R1 = int(alpha*L1*users/items)
    if(R2_option == 2):
        R2 = R1 * 2
    elif(R2_option == 3):
        R2 = R1 + 1
    else :
        R2 = users
    
    print("users->",users,":",L1,"-",L2)
    print("items->",items,":",R1,"-",R2)
    start_time = time.time()
    fin = SEAL(fileName, L, R1, R2)
    time_taken = (time.time() - start_time)
    
    
    fname = exp_type+"_instance"+str(instance)+"_L"+str(L)+"_epsilon"+str(epsilon)+"_R1"+str(R1)+"_R2"+str(R2)+"_alpha"+str(alpha)    
    directory=f"results/{exp_type}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(f"{directory}/{fname}.npy",fin) 
    
    
    users_alloc = np.zeros(v.shape[0])
    items_alloc = np.zeros(v.shape[1])
    utill_U = np.zeros(v.shape[0])
    utill_P = np.zeros(v.shape[1])

    for i in range(v.shape[0]):
        for j in fin[i]:
            users_alloc[i] += 1
            items_alloc[j] += 1
            utill_U[i] += v[i][j]
            utill_P[j] += v[i][j]
    
    print("algo: ",exp_type)
    print("instance: ",instance)
    print("L1: ",L1)
    print("L2: ",L1)
    print("R1: ",R1)
    print("R2: ",R2)
    print("log_Nash: ",sum(np.log(utill_U)))
    print("Revenue: ",np.sum(utill_U))
    print("Income Gap: ",np.max(utill_U)-np.min(utill_U))
    print("Time taken: ",time_taken)
    print("Minimum Utility: ",np.min(utill_U))
    print("Gini re-sellers: ",gini(utill_U))
    print("Gini Products: ",gini(utill_P))
    
    output2 = f"{exp_type},{instance},{L1},{L2},{R1},{users},{sum(np.log(utill_U))},{np.sum(utill_U)},{np.max(utill_U)-np.min(utill_U)},{time_taken},{alpha},{np.min(utill_U)},{gini(utill_U)},{gini(utill_P)},\n"
    result_file = f"results/{exp_type}.txt"              
    f= open(result_file,"a+")
    f.write(str(output2))
    f.close()
    
    
