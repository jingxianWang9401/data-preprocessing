# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:55:57 2020

@author: wangjingxian
"""

from sklearn import preprocessing
import numpy as np
import pandas as pd

x = np.array([[1., -1., 2., 3.],
              [2., 0., 0., -2],
              [0., 1., -1., 0],
              [1., 2., -3., 1]])


#data=pd.read_csv('E:/data_mining/v_a_dataset.csv')
#x=data.ix[:,0:2]
print("标准化之前的方差：", x.mean(axis=0))
print("标准化之前的标准差：", x.std(axis=0))

#标准化
x_scale = preprocessing.scale(x)
print("\n------------------\n标准化结果：\n", x_scale)
print("\n标准化之后的方差：", x_scale.mean(axis=0))
print("标准化之后的标准差：", x_scale.std(axis=0))


