# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:46:01 2020

@author: wangjingxian
"""

#1.缺失值比例（高：删！）
import pandas as pd
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
a = data_fea.isnull().sum()/len(data_fea)*100 #缺失值比例
cols = data_fea.columns# 列名
col = [ ]
for i in range(0,100):
    if a[i]<=50:   #缺失值阈值为50%
        col.append(cols[i])
print("缺失值低于阈值的特征共%d个；"%len(col),"\n它们分别是：", col)


#2.方差（低：删！）
import pandas as pd
import numpy as np
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值
#归一化
data_fea = data_fea.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
var = data_fea.var()#获取每一列的方差
cols = data_fea.columns
col = [ ]
for i in range(0,len(var)):
    if var[i]>=0.001:   # 将阈值设置为0.001
        col.append(cols[i])
print("高于阈值的特征共%d个；"%len(col),"\n它们分别是：", col)


#3、相关性矩阵（高：保留其一！）
import pandas as pd
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值
data_fea = data_fea.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#归一化
cor = data_fea[['x91', 'x92','x95','x96','x97',]].corr()#选取部分特征计算
# cor = data_fea.corr() #全部特征计算
print(cor)


#4、随机森林（低：删！）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
model = RandomForestRegressor(random_state=1, max_depth=10)
data_fea = data_fea.fillna(0)#随机森林只接受数字输入，不接受空值、逻辑值、文字等类型
data_fea=pd.get_dummies(data_fea)
model.fit(data_fea,data.y_增长率)
#根据特征的重要性绘制柱状图
features = data_fea.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:9])  # 因指标太多，选取前10个指标作为例子
plt.title('Index selection')
plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative importance of indicators')
plt.show()


#5、反向特征消除


#6、前向特征选择
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
datafile = u'data.csv'
data = pd.read_excel(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值
data_fea = data_fea.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))#归一化
fr = f_regression(data_fea,data.y_增长率)
cols = data_fea.columns
col = [ ]
for i in range(0,len(cols)-1):
    if fr[0][i] >=10:
        col.append(cols[i])
print("高于阈值的特征共%d个；"%len(col),"\n它们分别是：", col)

'''
高于阈值的特征共53个； 
它们分别是： ['x1', 'x2', 'x4', 'x5', 'x6', 'x7',
 'x9', 'x10', 'x11', 'x12', 'x13', 'x16', 'x18',
 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x26',
 'x27', 'x28', 'x34', 'x35', 'x36', 'x37', 'x38',
 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45',
 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52',
 'x53', 'x54', 'x55', 'x66', 'x67', 'x69', 'x70',
 'x71', 'x72', 'x73', 'x79', 'x80']
'''


#7、因子分析
#因子分析并不是通过删除特征来降低冗余数据的影响，而是通过挖掘潜在因子，在不删除特征的前提下降低影响。
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

#导入数据
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值
#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#因子分析，并选取潜在因子的个数为10
FA = FactorAnalysis(n_components = 10).fit_transform(data_fea.values)
#潜在因子归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
FA = min_max_scaler.fit_transform(FA)

#绘制图像，观察潜在因子的分布情况
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
plt.title('Factor Analysis Components')
plt.scatter(FA[:,0], FA[:,1])
plt.scatter(FA[:,1], FA[:,2])
plt.scatter(FA[:,2],FA[:,3])
plt.scatter(FA[:,3],FA[:,4])
plt.scatter(FA[:,4],FA[:,5])
plt.scatter(FA[:,5],FA[:,6])
plt.scatter(FA[:,6],FA[:,7])
plt.scatter(FA[:,7],FA[:,8])
plt.scatter(FA[:,8],FA[:,9])
plt.scatter(FA[:,9],FA[:,0])


#8、主成分分析-PCA
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#导入数据
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值

#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#选取主成分的个数为25
pca = PCA(n_components=25)
pca_result = pca.fit_transform(data_fea.values)

#绘制图像，观察主成分对特征的解释程度
plt.bar(range(25), pca.explained_variance_ratio_ ,fc='pink', label='Single interpretation variance')
plt.plot(range(25), np.cumsum(pca.explained_variance_ratio_),color='blue', label='Cumulative Explained Variance')
plt.title("Component-wise and Cumulative Explained Variance")
plt.legend()
plt.show()


#9、独立分量分析（ICA） 
#ICA寻找独立因素。如果两个特征是独立的，就代表两者之间没有任何关系。
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

#导入数据
datafile = u'data.csv'
data = pd.read_excel(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值

#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#独立分量分析（ICA）
ICA = FastICA(n_components=3, random_state=12) 
X=ICA.fit_transform(data_fea.values)

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#绘制图像，观察成分独立情况
plt.figure(figsize=(12,5))
plt.title('Factor Analysis Components')
plt.scatter(X[:,0], FA[:,1])
plt.scatter(X[:,1], FA[:,2])
plt.scatter(X[:,2],FA[:,0])


#10、manifold.Isomap算法
import numpy as np
import pandas as pd
from sklearn import manifold 
import matplotlib.pyplot as plt

#导入数据
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值

#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#降维
trans_data = manifold.Isomap(n_neighbors=5, n_components=3, n_jobs=-1).fit_transform(data_fea.values)
#n_neighbors：决定每个点的相邻点数,n_components：决定流形的坐标数,n_jobs = -1：使用所有可用的CPU核心

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
trans_data = min_max_scaler.fit_transform(trans_data)

#绘制图像
plt.figure(figsize=(12,5))
plt.scatter(trans_data[:,0], trans_data[:,1])
plt.scatter(trans_data[:,1], trans_data[:,2])
plt.scatter(trans_data[:,2], trans_data[:,0])


#11、manifold.TSNE算法
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#导入数据
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值

#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#降维
tsne = TSNE(n_components=3, n_iter=300).fit_transform(data_fea.values)

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
tsne = min_max_scaler.fit_transform(tsne)

#绘制图像
plt.figure(figsize=(12,5))
plt.scatter(tsne[:,0], tsne[:,1])
plt.scatter(tsne[:,1], tsne[:,2])
plt.scatter(tsne[:,2], tsne[:,0])


#12.UMAP算法
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

#导入数据
datafile = u'data.csv'
data = pd.read_csv(datafile)
data_fea = data.iloc[:,1:]#取数据中指标所在的列
data_fea = data_fea.fillna(0)#填补缺失值

#标准化
data_mean = data_fea.mean()
data_std = data_fea.std()
data_fea = (data_fea - data_mean)/data_std

#降维
umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3).fit_transform(data_fea.values)

#归一化
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
umap_data = min_max_scaler.fit_transform(umap_data)

#绘制图像
plt.figure(figsize=(12,5))
plt.scatter(umap_data[:,0], umap_data[:,1])
plt.scatter(umap_data[:,1], umap_data[:,2])
plt.scatter(umap_data[:,2], umap_data[:,0])








