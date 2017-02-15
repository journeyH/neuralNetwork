# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:59:04 2016
只能用于output_layer_size = 1
@author: hudie
"""

import numpy as np
import pandas as pd
import math
import scipy.optimize
import matplotlib.pyplot as plt
import nn_param

##设定参数
p = 9 #滞后阶数
input_layer_size  = p
hidden_layer_size = 5
init_epsilon = 0.12
#初始化输入层到隐藏层的权重矩阵
Theta1 = np.random.rand(hidden_layer_size,input_layer_size+1)*2*init_epsilon - init_epsilon
#初始化隐藏层到输出层的权重矩阵
Theta2 = np.random.rand(1,hidden_layer_size+1)*2*init_epsilon - init_epsilon
#向量化参数Theta1 Theta2
nn_params = np.r_[np.reshape(Theta1,[hidden_layer_size*(input_layer_size+1),1]), \
np.reshape(Theta2,[1*(hidden_layer_size+1),1])]
nn_params = nn_params.reshape(nn_params.shape[0]).tolist()
nn_lambda = 0.0#正则化参数


#载入数据
Original_data = np.loadtxt(open('/Users/hudie/Documents/py1/homework/zgjs.csv'))
#随机打乱
Original_data = np.random.permutation(Original_data)

#转化成Dataframe 主要是想用shift这一功能
Original_data_df = pd.DataFrame(Original_data)

for i in range(1,p+1):
    Original_data_df[i] = Original_data_df[i - 1].shift(-1) 
    
X = Original_data_df.as_matrix()
m, n = X.shape
X = X[: m - p] #删掉有nan的行
y = X[:,0]     
X = X[:,1:p+1] 
m,n = X.shape
#划分为训练集 交叉验证集 以及测试集
X_train = X[0:int(math.floor(m*0.5))]
y_train = y[0:int(math.floor(m*0.5))]
X_cv = X[int(math.floor(m*0.5)):int(math.floor(m*0.75))]
y_cv = y[int(math.floor(m*0.5)):int(math.floor(m*0.75))]
X_test = X[int(math.floor(m*0.75)):]
y_test = y[int(math.floor(m*0.75)):]

#得到要求的权重Theta1,Theta2
args_train=(input_layer_size,hidden_layer_size,X_train,y_train,nn_lambda)
args_cv=(input_layer_size,hidden_layer_size,X_cv,y_cv,nn_lambda)
args_test=(input_layer_size,hidden_layer_size,X_test,y_test,nn_lambda)
Theta =scipy.optimize.fmin_cg(nn_param.nnCostFunction,nn_params, fprime=nn_param.nnGrad,args=args_train,gtol=1e-10)
Theta1 = np.reshape(Theta[0:hidden_layer_size*(input_layer_size+1)],\
[hidden_layer_size,input_layer_size+1])
Theta2 = np.reshape(Theta[(hidden_layer_size*(input_layer_size+1)):],\
[1,hidden_layer_size+1])

#调试学习算法，根据训练集、交叉验证集合选择合适的lambda
kw = {'input_layer_size':input_layer_size,'hidden_layer_size':hidden_layer_size}
error_train,error_cv = nn_param.learningCurve(X_train,y_train,X_cv,y_cv,nn_lambda,nn_params,**kw)
#处理缺失
nanPositionET = ~np.isnan(error_train)
nanPositionEC = ~np.isnan(error_cv)
nanPosition = nanPositionET&nanPositionEC
error_train = error_train[nanPosition]
error_cv = error_cv[nanPosition]
#画出训练集合和交叉验证集的学习曲线

plt.figure(figsize = (7,4))
plt.plot(error_train,lw = 1.5,label = 'error_train')
plt.plot(error_cv,lw = 1.5, label = 'error_cv')
plt.legend(loc = 0)
plt.xlabel('number of  training examples')
plt.ylabel('error')
plt.title('learning curve for nn')

#得到测试集的误差
lambda0 = 0.0
#error_test
error_test=nn_param.nnCostFunction(Theta,kw['input_layer_size'],kw['hidden_layer_size'],X_test,y_test,lambda0)
print'error_test = %10.8f' %error_test









    



