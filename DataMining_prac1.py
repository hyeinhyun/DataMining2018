# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:18:37 2018

@author: Administrator
"""
# Use the following packages only
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 
from scipy.stats import stats
import matplotlib.pyplot as plt

def do_linear_regression(X,y):
    reg = LinearRegression()
    reg.fit(X,y)    
    return [reg.intercept_]+list(reg.coef_)

# Question 1
def predict(X, beta):
    ######## CALCULATE ESTIMATED TARGET WITH RESPECT TO X ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUPUT
    # y_pred: 1D list(array) with length n, the estimated target
    
    # TODO: prediction
    
    y_pred = []
    n,p=X.shape
    X=np.c_[np.ones(n),X]
    y_pred=np.matmul(X,beta)
    y_pred=y_pred.flatten()
    return y_pred

# Question 2
def cal_SS(X, y, beta):
    ######## CALCULATE SST, SSR, SSE ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # SST, SSR, SSE of the trained model
    
    # TODO: SS
    SST,SSR,SSE=0,0,0
    y_pred=predict(X,beta)
    SSR=sum((y_pred-np.mean(y))**2)
    SSE=sum((y-y_pred)**2)
    SST=sum((y-np.mean(y))**2)

    
    return (SST, SSR, SSE)


# Question 3
def f_test(X, y, beta, alpha):
    ######## PERFORM F-TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significant level
    # OUTPUT
    # f: f-test statistic of the model
    # pvalue: p-value of f-test
    # decision: f-test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: F-test
    f = 0
    pvalue = 0
    decision = None
    n,p=X.shape
    MSR=cal_SS(X,y,beta)[1]/p
    MSE=cal_SS(X,y,beta)[2]/(n-p-1)
    f=MSR/MSE
    pvalue=1-(fdist.cdf(f,p,n-p-1))
    
    if pvalue<alpha:
        decision=True
    else:
        decision=False
    return (f,pvalue,decision)

# Question 4
def cal_tvalue(X,y,beta):
    ######## CALCULATE T-TEST TEST STATISTICS OF ALL VARIABES ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    
    # TODO: t-test statistics
    
    SSE=cal_SS(X,y,beta)[2] #미리
    n,p=X.shape 
    MSE=SSE/(n-p-1)
    X=np.c_[np.ones(n),X] #1추가
    XtX=np.matmul(X.T,X)
    XtXinv=np.linalg.inv(XtX)
    t = []
    se_mat=MSE*XtXinv
    for i in range(p+1) :
        t.append(beta[i]/np.sqrt(se_mat[i,i]))
    return t

# Question 5
def cal_pvalue(t,X):
    ######## CALCULATE P-VALUE OF T-TEST TEST STATISTICS ########
    # INPUT
    # t: array(list) of size (k+1) with the t-test test statisc of variables (the first value is for intercept)
    # X: n by k (n=# of observations, k=# of input variables)
    # OUTPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    
    # TODO: p-value of t-test
    n,p=X.shape
    pvalue=[]
    for i in t:
        pvalue.append(1-(tdist.cdf(np.abs(i),n-p-1)))
    return pvalue

# Question 6
def t_test(pvalue,alpha):
    ######## DECISION OF T-TEST ########
    # INPUT
    # pvalue: array(list) of size (k+1) with p-values of t-test (the first value is for intercept)
    # alpha: significance level
    # OUTPUT
    # decision: array(list) of size (k+1) with t-test results of all variables
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    # TODO: t-test 
    decision = []
    for p in pvalue:
        if p<alpha/2:
            decision.append(True)
        else:
            decision.append(False)
    return decision

# Question 7
def cal_adj_rsquare(X,y,beta):
    ######## CACLULATE ADJUSTED R-SQUARE ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # OUTPUT
    # adj_rsquare: adjusted r-square of the model
    
    # TODO: adjusted r-square
    adj_rsquare=0 
    n,p=X.shape
    adj_rsquare=(1-(cal_SS(X,y,beta)[2]/(n-p-1))/(cal_SS(X,y,beta)[0]/(n-1)))
    return adj_rsquare

# Question 8
def skew(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # skew: skewness of the array x
    
    #TODO: calculate skewness
    #ONLY USE numpy
    skew = 0    
    s_mean=sum((x-np.mean(x))**3)/len(x)
    s_sd=(sum((x-np.mean(x))**2)/len(x))**(3/2)
    skew=s_mean/s_sd


    return skew

# Question 9
def kurtosis(x):
    ######## CACLULATE Skewness ########
    # INPUT
    # x: 1D list (array)
    # OUTPUT
    # kurt: kurtosis of the array x
    
    #TODO: calculate kurtosis
    #ONLY USE numpy
    kurt = 0
    k_mean=sum((x-np.mean(x))**4)/len(x)
    k_sd=(sum((x-np.mean(x))**2)/len(x))**2
    kurt=k_mean/k_sd
    return kurt

# Question 10
def jarque_bera(X,y,beta,alpha):
    ######## JARQUE-BERA TEST ########
    # INPUT
    # model: trained linear regression model
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # JB: Jarque-Bera test statistic
    # pvalue: p-value of the test statistic
    # decision: Jarque-Bera test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: Jarque-Bera test
    JB = 0
    pvalue =0
    n,p=X.shape
    print (n,p)
    y_pred=predict(X,beta)
    x=y-y_pred
    S=skew(x)
    C=kurtosis(x)
    JB=((n-p)/6)*(S**2+(1/4)*((C-3)**2))
    pvalue=1-chi2.cdf(JB,2)
    decision = None
    if pvalue < alpha :
        decision = True
    else :
        decision = False
    return (JB,pvalue,decision)

# Question 11
def breusch_pagan(X,y,beta,alpha):
    ######## BREUSCH-PAGAN TEST ########
    # INPUT
    # X: n by k (n=# of observations, k=# of input variables)
    # y: output target
    # beta: vector of estimated coefficient by do_linear_regression
    #       beta[0] is intercept
    #       the remained elements correspond to variables in X
    # alpha: significance level
    # OUTPUT
    # LM: Breusch-pagan Lagrange multiplier statistic
    # pvalue: p-value of the test statistics
    # decision: Breusch-pagan test result 
    #           True = reject null hypothesis
    #           False = accept null hypothesis
    
    # TODO: Breusch-pagan test
    LM = 0
    pvalue = 0
    decision = None
    n,p=X.shape
    y_pred = predict(X, beta)
    reside = y- y_pred
    reside_2 = reside**2
    beta_2 = do_linear_regression(X,reside_2)

    SST=cal_SS(X,reside_2,beta_2)[0]
    SSR=cal_SS(X,reside_2,beta_2)[1]
    
    R_2 = SSR/SST  
    LM = n * R_2 
    pvalue = 1-chi2.cdf(LM,p-1)
    
    if pvalue < alpha:
        decision = True
    else:
        decision = False
    return (LM,pvalue,decision)


if __name__=='main':
    # LOAD DATA
    data = pd.read_csv('https://drive.google.com/uc?export=download&id=1YPnojmYq_2B_lrAa78r_lRy-dX_ijpCM', sep='\t')
    # INPUT
    X = data[data.columns[:-1]]
    # TARGET
    y = data[data.columns[-1]]
    alpha = 0.05
    coefs= do_linear_regression(X,y)
    #################### TEST YOUR CODE ####################
    print(predict(X,coefs))
    print(cal_SS(X,y,coefs))
    print(f_test(X,y,coefs,alpha))
    print(cal_tvalue(X,y,coefs))
    t = cal_tvalue(X,y,coefs)
    cal_pvalue(t,X)
    p = cal_pvalue(t,X)
    t_test(p,alpha)
    
    print(cal_adj_rsquare(X,y,coefs))
    
    x = y-predict(X, coefs)
    print(skew(x))
    print(kurtosis(x))
    
    print(jarque_bera(X,y,coefs,alpha))
    print(breusch_pagan(X,y,coefs,alpha))
