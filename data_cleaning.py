# -*- coding: utf-8 -*-
"""
Created on Fri May 22 08:32:47 2020

@author: wangjingxian
"""


import pandas as pd


#读取文件
datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv



#表格中缺失值的处理操作，只要该行有缺失值，那么就删除该行数据（一条样本）
'''
data.dropna() #直接删除含有缺失值的行
data.dropna(axis = 1) #直接删除含有缺失值的列
data.dropna(how = 'all') #只删除全是缺失值的行
data.dropna(thresh = 3) #保留至少有3个非空值的行
data.dropna(subset = [u'血型 ']) #判断特定的列，若该列含有缺失值则删除缺失值所在的行
'''
print("显示缺失值，缺失则显示为TRUE：\n", data.isnull())#是缺失值返回True，否则返回False
print("---------------------------------\n显示每一列中有多少个缺失值：\n",data.isnull().sum())#返回每列包含的缺失值的个数
data.dropna(axis=0, how='any', inplace=True)
'''
axis：0-行操作（默认），1-列操作 
how：any-只要有空值就删除（默认），all-全部为空值才删除 
inplace：False-返回新的数据集（默认），True-在愿数据集上操作
'''
print(data)


#插补缺失值，可以使用常用的简单缺失值插补方法对表格中的空值进行填充
'''
data.fillna(data.mean())  #均值插补
data.fillna(data.median()) #中位数插补
data.fillna(data.mode())  #众数插补
data.fillna(data.max())   #最大值插补
data.fillna(data.min())   #最小值插补
data.fillna(0)         #固定值插补--用0填充
data.fillna(5000)       #固定值插补--用已知的行业基本工资填充
data.fillna（method='ffill'）#最近邻插补--用缺失值的前一个值填充
data.fillna（method='pad'） #最近邻插补--用缺失值的前一个值填充
'''
datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv
print("------------------\n用均值插补后的数据data：\n", data.fillna(data.mean()))


#对于缺失数据采用：拉格朗日插值法
#结果不对，拉格朗日函数最后求解出来的是负数，与实际结果不符，这个问题待解决！！！！！！
import pandas as pd
from scipy.interpolate import lagrange#拉格朗日函数
data=pd.read_csv('E:/data_mining/data_preprocessing/data/data1.csv')

#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ploy(s,n,k=5):
    y=s[list(range(n-k,n))+list(range(n+1,n+1+k))]#取数
    y=y[y.notnull()] #剔除空值
    return lagrange(y.index,list(y))(n)#返回拉格朗日函数结果

#逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if(data[i].isnull())[j]:
            data[i][j]=ploy(data[i],j)
print('空白处插补之后的数据为：',data)
data.to_csv('E:/data_mining/data_preprocessing/data/data2.csv')



#表格中含有特殊字符，将含有特殊字符的整行进行删除
datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置
data = pd.read_csv(datafile)#如果是csv文件则用read_csv
df1=pd.DataFrame(data,columns=list('va'))
data=df1[ ~ data['a'].isin([1])]#不加~是选取表格中含有1的行，加~是取反，删除表格中含有1的行
print('删除特殊字符后的数据为：',data)
#data.to_csv('E:/data_mining/data_preprocessing/data/data2.csv',index=False) #将数据重新写入excel


#重复值处理
#在Pandas中，.duplicated()表示找出重复的行，默认是判断全部列，返回布尔类型的结果。
#对于完全没有重复的行，返回 False，对于有重复的行，第一次出现的那一行返回 False，其余的返回 True。
#与.duplicated()对应的，.drop_duplicates()表示去重，即删除布尔类型为 True的所有行，默认是判断全部列。
import pandas as pd
import numpy as np
from pandas import DataFrame,Series
#读取文件
datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_csv(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
#去重
print(examDf.duplicated())#判断是否有重复行，重复的显示为TRUE，
examDf.drop_duplicates()#去掉重复行
print('去重',examDf.drop_duplicates())


datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_csv(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
#指定某列判断是否有重复值
print(examDf.duplicated('功率'))#判断name列是否有重复行，重复的显示为TRUE，
examDf.drop_duplicates('功率')#去掉重复行
print('删除某列重复',examDf.drop_duplicates('功率'))


datafile = u'E:/data_mining/data_preprocessing/data/data1.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_csv(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
#根据多列判断是否有重复值
print(examDf.duplicated(['v','a']))#判断name,sex,birthday列是否有重复行，重复的显示为TRUE，
examDf.drop_duplicates(['v','a'])#去掉重复行
print('删除多列重复',examDf.drop_duplicates(['v','a']))



#异常值进行处理
import pandas as pd #导入pandas库
inputfile = u'E:/data_mining/data_preprocessing/data/data1.csv'
data= pd.read_csv(inputfile)
#将功率因数低于70或者高于95的异常值清空
data[u'power_factor'][(data[u'power_factor']<70) | (data[u'power_factor']>95)] = None 
#清空后删除
print(data.dropna())

#将功率因数低于70或者高于95的异常值清空
inputfile = u'E:/data_mining/data_preprocessing/data/data1.csv'
data= pd.read_csv(inputfile)
data[u'power_factor'][(data[u'power_factor']<70) | (data[u'power_factor']>95)] = None 
#清空后用均值插补
print(data.fillna(data.mean()))


