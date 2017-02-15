# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:13:35 2016

@author: hudie
"""
import numpy as np
import pandas as pd
import scipy.optimize

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,X,y,nn_lambda):
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],\
    [hidden_layer_size,input_layer_size+1])
    Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):],\
    [1,hidden_layer_size+1])   
    m,n = X.shape
    X = np.c_[np.ones(m),X] #加入偏置  
    #forward propagation
    z2 = np.dot(X,np.transpose(Theta1))
#   隐藏层为线性  
#   a2 = z2
#   隐藏层为sigmoid
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m),a2]
    z3 = np.dot(a2,np.transpose(Theta2))
    hypo = z3
    #sigmoid
    #hypo = sigmoid(z3)
    #没有正则化的代价函数
    #sigmoid
    #J = (1/m)*(-np.dot(y,np.log(hypo)) - np.dot((1-y),np.log(1-hypo)))  
    #linear
    J = (1.0/(2.0*m))*np.sum((hypo - y)**2.0)
    temp1 = Theta1.copy()
    temp1[:,0] = 0.0
    temp2 = Theta2.copy()
    temp2[:,0] = 0.0
    penal_J = nn_lambda * (np.sum(temp1 ** 2.0)+np.sum(temp2 **2.0))/(2.0*m)
    #正则化代价函数
    J = J + penal_J  
    return J


def nnGrad(nn_params,input_layer_size,hidden_layer_size,X,y,nn_lambda):    
    Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size+1)],\
    [hidden_layer_size,input_layer_size+1])
    Theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):],\
    [1,hidden_layer_size+1]) 
    m,n = X.shape
    X = np.c_[np.ones(m),X] #加入偏置  
    #forward propagation
    z2 = np.dot(X,np.transpose(Theta1))
#   隐藏层为线性    
#   a2 = z2
#   隐藏层为sigmoid
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m),a2]
    z3 = np.dot(a2,np.transpose(Theta2))
#   hypo = sigmoid(z3)    
    hypo = z3
    #Back propogation
    m1,n1 = np.shape(Theta1)
    m2,n2 = np.shape(Theta2)
    Delta_1 = np.zeros([m1,n1])
    Delta_2 = np.zeros([m2,n2])
    for i in range(m):
        delta_3 = hypo[i] - y[i]
        delta_2 = np.transpose(Theta2) * delta_3
        delta_2 = delta_2[1:]
        delta_2 = delta_2 * sigmoidGradient(z2[i]).reshape(hidden_layer_size,1)
        Delta_2 = Delta_2 + delta_3 * a2[i]
        Delta_1 = Delta_1 + np.dot(delta_2,X[i].reshape(1,n+1))
    #没有正则化的梯度
    Theta1_grad = (1.0/m)*Delta_1
    Theta2_grad = (1.0/m)*Delta_2
    
    temp1 = Theta1.copy()
    temp1[:,0] = 0.0
    temp2 = Theta2.copy()
    temp2[:,0] = 0.0
    penel_Grad1 = (nn_lambda/m)*temp1
    penel_Grad2 = (nn_lambda/m)*temp2
    #正则梯度
    Theta1_grad = Theta1_grad + penel_Grad1
    Theta2_grad = Theta2_grad + penel_Grad2
    grad = np.r_[np.reshape(Theta1_grad,[hidden_layer_size*(input_layer_size+1),1]), \
    np.reshape(Theta2_grad,[1*(hidden_layer_size+1),1])]
    mgrad,ngrad = np.shape(grad)
    grad = np.reshape(grad,[1,mgrad])
    return grad.reshape(grad.shape[1])
    
def learningCurve(X_train,y_train,X_cv,y_cv,nn_lambda,nn_params,**kw):
    m,n = X_train.shape
    error_train = np.zeros([m,1])
    error_cv = np.zeros([m,1])
    lambda0 = 0.0   
    for i in range(1,m):
        args_train=(kw['input_layer_size'],kw['hidden_layer_size'],X_train[0:i],y_train[0:i],nn_lambda)
        theta = scipy.optimize.fmin_cg(nnCostFunction,nn_params, fprime=nnGrad,args=args_train,gtol=1e-10)
        error_train[i] = nnCostFunction(theta,kw['input_layer_size'],kw['hidden_layer_size'],X_train[0:i],y_train[0:i],lambda0)
        error_cv[i] = nnCostFunction(theta,kw['input_layer_size'],kw['hidden_layer_size'],X_cv,y_cv,lambda0)
    return error_train,error_cv

def sigmoid(z):
    g = 1.0 / (1.0 +np.exp(-z))
    return g
    
def sigmoidGradient(z):
    g = sigmoid(z) * (1.0 - sigmoid(z))
    return g
    

    