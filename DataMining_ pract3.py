# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:05:51 2018

@author: Administrator
"""
# Use the following packages only
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from scipy.misc import comb
from itertools import combinations
import matplotlib.pyplot as plt

# QUESTION 1
def gini(D):
    ######## GINI IMPURITY ########
    # INPUT 
    # D: 1-D array containing different classes of samples
    # OUTPUT    
    # gini: Gini impurity
    #n,p=D.shape
    #print(D)
    if len(D) !=0:
        count_M=0
        count_B=0
        for i in D:
            if i=='M':
                count_M=count_M+1
            else:
                count_B=count_B+1
        
        # TODO: Gini impurity
        gini=1-np.square(count_M/len(D))-np.square(count_B/len(D))
    else:
        gini=1
    return gini

# QUESTION 2
def split_gini(x,y):

    ######## FIND THE BEST SPLIT ########
    # INPUT 
    # x: a input variable 
    Gini_before=gini(y)
    split_point=0    
    gain =0
    IG=[]
    sorted_list = pd.DataFrame({'x':x,'y':y})
    sorted_list=sorted_list.sort_values(by='x')
    sorted_list =sorted_list.reset_index(drop=True)
    x=sorted_list['x'].values
    y=sorted_list['y'].values

    for i in range(len(y)):
        Info_gain=Gini_before-gini(y[:i])*(i+1/len(x))-gini(y[i:])*(1-(i+1/len(x)))
        IG.append(Info_gain)
    gain=max(IG)
    index=IG.index(gain)
    split_point=(x[index]+x[index+1])/2
    
    
        

                
    # y: a output variable
    # OUTPUT
    # split_point: a scalar number for split. Split is performed based on x>split_point or x<=split_point
    # gain: information gain at the split point
    
    # TODO: find the best split

    return (split_point,gain)


# QEUSTION 3
def kmeans(X,k,max_iter=300):
    ############ K-MEANS CLUSTERING ##########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of clusters
    # max_iter: the maximum number of iteration
    # OUTPUT
    # label: cluster label (len(label)=n)
    # centers: cluster centers (k by p)
    ##########################################
    # If average distance between old centers and new centers is less than 0.000001, stop
    n,p=X.shape
    # TODO: k-means clustering
    
    centers = []
    #selct inital centroid (index)
    center = np.random.randint(0,n,k)
    for i in range(len(center)):
        centers.append(X[i])
    
    #comparing distance
    for z in range(max_iter):
        label = []
        center_temp=[0,0,0]
        for i in X:
            compare=[]
            ### i=a.index("t")
            for j in range(k):
                c=euclidean_dist(i,centers[j])
                compare.append(c)
            index=compare.index(min(compare))
            label.append(index)#애가 속한 cluster
        #print(label)
        #센터 다시 구하기
        
        for i in range(k):
            #temp=np.array([])
            #임시로 같은 클러스터에 있는 애들 모아줌
            temp = []

            for j in range(len(label)):
                if label[j]==i:
                    temp.append(X[j].tolist())
            temp_arr=np.array(temp)
            center_temp[i]=temp_arr.mean(axis=0)
        #print(center_temp)
        #print(center)#새로운 센터를 찾는다.
        if np.array_equal(center_temp,centers):
            center=center_temp
            print(z)
            print('stop')
            break
        else:
            centers=center_temp

    
   # center=center_temp
    

    
    return (label, centers)

# QUESTION 4
def cal_support(data,rule):
    ######## CALCULATE SUPPORT OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # support: support of the rule
    #######################################################
    count=0
    for x in data:
        if all(i in x for i in rule[0]):
            count=count+1
    
    # TODO: support 
    support = 0
    support=count/len(data)
    return support

# QUESTION 5
def cal_conf(data,rule):
    ######## CALCULATE CONFIDENCE OF ASSOCIATION RULE ########
    # INPUT
    # data: transaction data, each row contains items
    # rule: array or list with two elements, rule[0] is a set of condition items and rule[1] is a set of result itmes
    # OUTPUT
    # confidence: confidence of the rule
    #########################################################
    support=cal_support(data,rule)
    count=0
    for x in data:
        if all(i in x for i in rule[0]):
            if all (j in x for j in rule[1]):
                count=count+1
    # TODO: confidence
    confidence = 0
    confidence=count/(support*len(data))
    return confidence

# QEUSTION 6
def generate_ck(data,k,Lprev=[]):
    ######## GENERATE Ck ########
    # INPUT
    # data: transaction data, each row contains items
    # k: the number of items in sets
    # Lprev: L(k-1) for k>=2
    # OUTPUT
    # Ck: candidates of frequent items sets with k items
    ##############################
    
    # TODO: Ck
    if k==1:
        Ck=[]
        Ck=set(sum(data,[]))
        return Ck
    else:
        Ck=[]
        Lprev_list=[]
        for i in Lprev:
            for j in i:
                Lprev_list.append(list(j))
        print(Lprev)
        ck_prev=set(sum(Lprev_list,[]))
        #print('******')
        #print(ck_prev)
        #print('******')

        for c in combinations(ck_prev,k):
            print('******')
            print(c)
            #print(combinations(c,k-1))
           # for x in combinations(c,k-1):
            #    print(x)
            if all(x in Lprev for x in combinations(c,k-1)):
               Ck.append(c)
        

        
        return Ck

# QEUSTION 7
def generate_lk(data,Ck,min_sup):
    ######## GENERATE Lk ########
    # INPUT
    # data: transaction data, each row contains items
    # Ck: candidates of frequent items sets with k items
    # min_sup: minimum support
    # OUTPUT
    # Lk: frequent items sets with k items
    ##############################
    
    # TODO: Lk
    # Use cal_support
    Lk=[]
    for k in Ck:
        lk=[]
        check=cal_support(data,k)
        if check>min_sup:
            Lk.append(tuple(k))
    
    return Lk
# QEUSTION 8
def PCA(X,k):
    ######## PCA ########
    # INPUT
    # X: n by p array (n=# of observations, p=# of input variables)
    # k: the number of components
    # OUTPUT
    # components: p by k array, each column corresponds to PC in order. (the first PC is the first column)
    
    # TODO: PCA
    # Hint: use numpy.linalg.eigh
    
    X_matrix = X-np.mean(X, axis=0)
    X_cov = np.matmul(X_matrix.T,X_matrix)
    eigen_value, eigen_vector = np.linalg.eigh(X_cov,UPLO='L')
    print(eigen_vector)
    
    t=np.argsort(-eigen_value) #eigen_value 로 sort
    components = eigen_vector[:,t[0:k]] #큰애들 추출 (k개)


    return components


def euclidean_dist(a,b):
    d = 0
    a=np.array(a)[0]
    b=np.array(b)[0]
    
    d= np.sqrt(np.sum((a-b)**2))
    
    return d

    
if __name__=='main':    
    cancer=pd.read_csv('https://drive.google.com/uc?export=download&id=1-83EtpdXI_bNWlWD7v-t_7XLJgnwocxg')
    #Value

    iris=load_iris()
    trans=pd.read_csv('https://drive.google.com/uc?export=download&id=1F_6wOpWqO-yXfbpfSCXfX6_uV4YhOPqD', index_col=0)
    trans=[x.split(',') for x in trans['Items'].values]
    
    #################### TEST YOUR CODE ####################
###for Question 1,2
    X = cancer[cancer.columns[2:12]]
    # TARGET
    y = cancer[cancer.columns[1]]
    for x in X:
        print(split_gini(cancer[x],y))
        
        
#for Question 3
    X=iris.data
    y=iris.target
    X_select=X[:,[0,1]]
    X_select.mean(axis=0)
    data=kmeans(X_select,3)
    l=data[0]
    #실험결과
    plt.scatter(X[:,0],X[:,1],c=l)
    plt.scatter(X[:,0],X[:,1],c=y)

#for Question 4,5
    cal_support(trans,['a','b'])
    cal_conf(trans,['a','b'])   
    cal_support(trans,[['b','c','e'],'f'])     
    cal_conf(trans,[['b','c','e'],'f'])   
    cal_support(trans,[['a','c'],['b','f']])     
    cal_conf(trans,[['a','c'],['b','f']])     
    cal_support(trans,[['b','d'],'g'])     
    cal_conf(trans,[['b','d'],'g'])     
    cal_support(trans,[['b','e'],['c','f']])     
    cal_conf(trans,[['b','e'],['c','f']])     

  


#for Q8
    X=iris.data
    y=iris.target
    #X_select=X[:,[0,1]]
    y=PCA(X,2)
    
    result=np.matmul(X,y)
    X_t=result[:,0]
    plt.scatter(result[:,0],result[:,1],c=iris.target)
    
    
    
    
    # Apriori algorithm
    min_sup=0.4    
    Ck=generate_ck(trans,1)
    Ck
    r=dict()
    for k in range(1,len(Ck)):
        Lk=generate_lk(trans,Ck,min_sup)
        r[k]=[Ck,Lk]
        Ck=generate_ck(trans,k+1,Lk)
        if len(Ck)==0:
            break