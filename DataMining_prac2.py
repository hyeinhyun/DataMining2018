# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:47:40 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def logistic_reg(X,y):
    clf=LogisticRegression()
    clf.fit(X,y)
    return list(clf.intercept_)+list(clf.coef_[0])

# QUESTION 1
def cal_logistic_prob(X,y,beta):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target (Assumption: binary classification problem)
    # beta: array(list) of size (k+1) with the estimated coefficients of variables (the first value is for intercept)
    #        This coefficiens are for P(Y=k) where k is the larger number in output target variable
    # OUTPUT
    # p: probability of P(Y=k) where k is the larger number in output target variable
    
    # TODO: calculate proability of the class with respect to the given X for logistic regression
    p=[]
    n=X.shape[0]
    X=np.c_[np.ones(n),X]
    t=np.matmul(X,beta)
    print(t)
    p=1/(1+np.exp(-t))

    return p

# QUESTION 2
def cal_logistic_pred(y_prob,cutoff,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # cutoff: threshold for decision
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob and cutoff (logistic regression)
    # if probability>cutoff → classes[1] else classes [0]
    y_pred=[]
    n=len(y_prob)
    for i in range(n):
        if y_prob[i]>cutoff:
            y_pred.append(classes[1])
        else:
            y_pred.append(classes[0])
    return y_pred

# QUESTION 3    
def cal_acc(y_true,y_pred):
    ######## CALCULATE ACCURACY ########
    # INPUT
    # y_true: array(list), true class
    # y_pred: array(list), estimated class
    # OUPUT
    # acc: accuracy
    
    # TODO: calcuate accuracy
    n=len(y_true)
    count=0
    for i in range(n):
        if y_true[i]==y_pred[i]:
            count=count+1
    
    acc = 0
    acc=count/n    
    return acc

# QUESTION 4   
def BNB(X,y):
    ######## BERNOULLI NAIVE BAYES ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # y: output (len(y)=n, categorical variable)
    # OUTPUT
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
        
    # TODO: Bernoulli NB
    n,p=X.shape
    X_ber=[]
    avg_arr=[]
    class_1=[]
    class_2=[]
    pmatrix=[]    
    pT=[]    

    X_ber=(X>np.mean(X,axis=0))*1  #binary화 시킨것!!
    #class1개수/class2인 개수 카운트 --> 각 p속성들에서 x=1인 개수 세
    num_c1=0
    num_c2=0
    for i in range(n):
        if y[i]==1:
            num_c1=num_c1+1
        else:
            num_c2=num_c2+1
    print(num_c1)
    print(num_c2)
            
    #X에서 class1인 애들/class2인 애들
    for j in range(p):
        count_c1=0
        count_c2=0
        for i in range(n):
            if y[i]==1:
                if int(X_ber[i][j])==1:
                    count_c1=count_c1+1
            else:
                if int(X_ber[i][j])==1:
                    count_c2=count_c2+1
        print(count_c1)
        print(count_c2)
        class_1.append(count_c1/num_c1)
        class_2.append(count_c2/num_c2)
    pT.append(class_1)
    pT.append(class_2)
    print(pT)
    
    pmatrix=np.array(pT).T
    print(pmatrix)
    
    
    return pmatrix

# QUESTION 5
def cal_BNB_prob(X,prior,pmatrix):
    ######## CALCULATE PROBABILITY OF THE CLASS ########
    # INPUT 
    # X: n by p array (n=# of observations, p=# of input variables)
    # priors: 1D array of size c where c is number of unique classes in y, prior probabilities for classes
    # pmatrix: 2-D array(list) of size c by p with the probability p_ij where c is number of unique classes in y
    # OUTPUT
    # p: n by c array, p_ij stores P(y=cj|X_i)
    n,t=X.shape
    X_ber=[]    
    avg_arr=[]
    class_1=[]
    class_2=[]

    X_ber=(X>np.mean(X,axis=0))*1 
    
    #베르누이 각 샘플별 게산
    for i in range(n):
        c_1=1
        c_2=1
        for j in range(t):
            c_1=c_1*(pmatrix[j][0]**X_ber[i][j])*((1-pmatrix[j][0])**(1-X_ber[i][j]))
            c_2=c_2*(pmatrix[j][1]**X_ber[i][j])*((1-pmatrix[j][1])**(1-X_ber[i][j]))
        c_1=c_1*prior[0]
        c_2=c_2*prior[1]
        class_1.append(c_1)
        class_2.append(c_2)
        
    # TODO: calculate proability of the class with respect to the given X for Bernoulli NB
    pT=[]
    pT.append(class_1)
    pT.append(class_2)
    p=np.array(pT).T
    return p

# QUESTION 6
def cal_BNB_pred(y_prob,classes):
    ######## ESTIMATE OUTPUT CLASS ########
    # INPUT
    # y_prob: probability of P(Y=k) where k is the larger number in output target variable
    # classes: labels of classes
    # OUTPUT
    # y_pred: array(list) with the same size of y_prob, estimated output class 
    
    # TODO: estimate output class based on y_prob (Bernoulli NB)
    n=len(classes)  
    s=len(y_prob)
    y_pred=[]
    print(s)
    print(y_prob)

    for i in y_prob:
        if i[0]>i[1]:
            y_pred.append(1)
        else:
            y_pred.append(2)
    print(y_pred)
            
        
    return y_pred

    
# QUESTION 7
def euclidean_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Euclidean distance between a and b
    
    # TODO: Euclidean distance
    d_2=0
    
    n=len(a)
    for i in range(n):
        d_2=d_2+(a[i]-b[i])**2
    d=np.sqrt(d_2)
    # TODO: Euclidean distance
    return d

# QUESTION 8
def manhattan_dist(a,b):
    ######## EUCLIDEAN DISTANCE ########
    # INPUT
    # a: 1-D array 
    # b: 1-D array 
    # a and b have the same length
    # OUTPUT
    # d: Manhattan distance between a and b
    
    # TODO: Manhattan distance
    n=len(a)
    # TODO: Manhattan distance
    d = 0
    for i in range(n):
        d=d+np.abs(a[i] - b[i])
    return d

# QUESTION 9
def knn(trainX,trainY,testX,k,dist=euclidean_dist):
    ######## K-NN Classification ########
    # INPUT 
    # trainX: training input dataset, n by p size 2-D array
    # trainY: training output target, 1-D array with length of n
    # testX: test input dataset, m by p size 2-D array
    # k: the number of the nearest neighbors
    # dist: distance measure function
    # OUTPUT
    # y_pred: predicted output target of testX, 1-D array with length of m
    #         When tie occurs, the final class is select in alpabetical order
    #         EX) if "A" ties "B", select "A" and if "2" ties "4", select 2
    
    # TODO: k-NN classification
    
    y_pred=[]
    m,p=testX.shape
    print(m)
    n=trainX.shape[0]
    print(n)

    for i in range(m):
            y_dist=[]
            for j in range(n):
                dist_1=[]
                d=dist(trainX[j][:],testX[i][:])
                dist_1.append(d)
                dist_1.append(j)
                y_dist.append(dist_1)
            y_dist=sorted(y_dist,key=lambda x:x[0])
            
            check1=0
            check2=0
            for t in range(k):
                #a.append(y_dist[t])
                a=y_dist[t]
                b=a[1]
                if trainY[b]==1:
                    check1=check1+1
                else:
                    check2=check2+1
            if check1>check2:
                y_pred.append(1)
            else:
                y_pred.append(2)
                
            
        
    
                
            #sorted(student_tuples, key=lambda student: student[2])
            
    return y_pred






def prior_t(y):
    n=len(y)
    n_1=0
    n_2=0
    for i in range(n):
        if y[i]==1:
            n_1=n_1+1
        else:
            n_2=n_2+1
    prior=[]
    prior.append(n_1/n)
    prior.append(n_2/n)
    return prior
def line(t_y,e_y):
    y_axis=[]
    r=np.arange(0.1,1,0.05)
    for i in r:
        p=cal_logistic_pred(e_y,i,[1,2])
        a=cal_acc(t_y.values,p)
        y_axis.append(a)
    print(y_axis)
    plt.plot(r,y_axis)
    plt.show()

if __name__=='main':
    data=pd.read_csv(r'https://drive.google.com/uc?export=download&id=1QhUgecROvFY62iIaOZ97LsV7Tkji4sY4',names=['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Y'])
    y=data['Y']
    X=data.loc[(y==1)|(y==2),['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
    y=y.loc[(y==1)|(y==2)]
    
    trainX,testX,trainY,testY=train_test_split(X,y,test_size=0.2,random_state=11)
    #################### TEST YOUR CODE ####################
    #logistic
    beta=logistic_reg(X,y)
    y_prob=cal_logistic_prob(X,y,beta)
    line(y,y_prob)
    y_pred=cal_logistic_pred(y_prob,0.5,[1,2])
    cal_acc(y.values,y_pred)
    pmatrix=BNB(X.values,y.values)
    prior=prior_t(y.values)

    y_prob=cal_BNB_prob(X.values,prior,pmatrix)
    print(len(y_prob))
    print(y_prob[0])
    y_pred=cal_BNB_pred(y_prob,[1,2])
    cal_acc(y.values,y_pred)
    a=knn(trainX.values,trainY.values,testX.values,3,dist=euclidean_dist)
    cal_acc(testY.values,a)
    a=knn(trainX.values,trainY.values,testX.values,5,dist=euclidean_dist)
    cal_acc(testY.values,a)
    a=knn(trainX.values,trainY.values,testX.values,7,dist=euclidean_dist)
    cal_acc(testY.values,a)
    a=knn(trainX.values,trainY.values,testX.values,3,dist=manhattan_dist)
    cal_acc(testY.values,a)
    a=knn(trainX.values,trainY.values,testX.values,5,dist=manhattan_dist)
    cal_acc(testY.values,a)
    a=knn(trainX.values,trainY.values,testX.values,7,dist=manhattan_dist)
    cal_acc(testY.values,a)

    
    
    
    
    #naiive bayes
    
    
    
    
    
    
    