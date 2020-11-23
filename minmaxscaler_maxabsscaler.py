# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:29:58 2020

@author: wangjingxian
"""
#数据归一化到[0,1][-1,1]

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler 


#eg：将数据归一到 [ 0，1 ] 
data=pd.read_csv('E:/data_mining/v_a_dataset.csv')
x=data.ix[:,0:2]
min_max_scale=MinMaxScaler().fit(x)#训练规则
x_minmax=min_max_scale.transform(x)#应用规则
print(x_minmax)

#如果有新的测试数据进来，也想做同样的转换，那么将新的测试数据添加到原数据末尾即可
data=pd.read_csv('E:/data_mining/v_a_dataset.csv')
x=data.ix[:,0:2]
data_new=pd.read_csv('E:/data_mining/data_preprocessing/data/data1.csv')
y=x=data.ix[:,0:2]
x.append(y)#将y添加到x的末尾
print('x ：\n', x)
min_max_scale=MinMaxScaler().fit(x)#训练规则
x_minmax=min_max_scale.transform(x)#应用规则
print('x_minmax :\n', x_minmax)


#MaxAbsScaler：归一到 [ -1，1 ] 
data=pd.read_csv('E:/data_mining/v_a_dataset.csv')
x=data.ix[:,0:2]
max_abs_scale=MaxAbsScaler().fit(x)#训练规则
x_maxabs=max_abs_scale.transform(x)#应用规则
print(x_maxabs)

#如果有新的测试数据进来，也想做同样的转换，那么将新的测试数据添加到原数据末尾即可
data=pd.read_csv('E:/data_mining/v_a_dataset.csv')
x=data.ix[:,0:2]
data_new=pd.read_csv('E:/data_mining/data_preprocessing/data/data1.csv')
y=x=data.ix[:,0:2]
x.append(y)#将y添加到x的末尾
print('x ：\n', x)
max_abs_scale=MaxAbsScaler().fit(x)#训练规则
x_maxabs=max_abs_scale.transform(x)#应用规则
print('x_maxabs :\n', x_maxabs)

