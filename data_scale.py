# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:01:36 2020

@author: wangjingxian
"""

#利用pandas和numpy对数据进行操作，使用matplotlib进行图像化，使用sklearn进行数据集训练与模型导入。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("v_a_dataset.csv")#数据集读取，csv格式
'''
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

data[['回路电压']].apply(max_min_scaler)

'''

'''
另外一种数据归一化的方法
from scipy.cluster.vq import whiten
#将原始数据做归一化处理
data_scale=whiten(data)

print('归一化后的数据为',data_scale)
'''

data = (data-data.min())/(data.max()-data.min())#即简单实现标准化
print(data)

data.to_csv(path_or_buf='v_a_guiyihua.csv')


'''
#规格化
scale = (data['a'] - data['a'].min())/(data['a'].max() - data['a'].min())

#安全删除，如果用del是永久删除

df2 = data.drop(['a'],axis=1)

#把规格化的那一列插入到数组中,最开始的14是我把他插到了第15lie

df2.insert(2,'a',scale)

print(df2.columns[1:2])
'''