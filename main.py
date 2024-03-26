# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:31:42 2020

@author: yryi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report


'''
处理数据
'''
#读取数据
data_train = pd.read_excel('./NonInvasiveFetalECGThorax1_TRAIN.xls',header=None)
data_test = pd.read_excel('./NonInvasiveFetalECGThorax1_TEST.xls',header=None)
#设置数据列名
cols = ['label']
for i in range(255):
    cols_str = 't'+str(i+1)
    cols.append(cols_str)
data_train.columns = cols
data_test.columns = cols
# 标签数组
lable_arry  = data_train.label.unique()
label_len = lable_arry.shape[0]
# 训练集可视化
for i in range(label_len):
    data_temp = data_train[data_train['label']==i+1].drop(['label'],axis=1).values
    n = data_temp.shape[0]
    plt.figure()
    plt.title(f'label_{i+1}')
    for j in range(n):
        plt.plot(data_temp[j,:])
    plt.savefig(f'lable_{i+1}')

'''
模型训练与测试
'''
# 特征与目标值
X_train = data_train.drop('label',axis=1).values
y_train = data_train['label'].values

# 测试集随机抽样
ration = 0.6
data_test = data_test.sample(frac=ration,random_state=1)
X_test = data_test.drop('label',axis=1).values
y_test = data_test['label'].values
#PAC降维
pca = PCA(n_components=10,random_state=2020)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# 特征标准化
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.fit_transform(X_test)
#训练模型
rf = RandomForestClassifier(n_estimators=400, criterion='gini',\
                       oob_score=True, n_jobs=-1, random_state=2020,\
                      verbose=1, class_weight='balanced')
rf.fit(X_train, y_train)
#测试集上预测
y_pred = rf.predict(X_test)


'''
模型评价
'''

#输出多分类评价指标
accuracy = rf.score(X_test, y_test)
print(f'Classification accuracy: {accuracy}')
fmeasure_weight = metrics.f1_score(y_test, y_pred, average='weighted')
print (f'Weighted F-measure: {fmeasure_weight}')
fmeasure_micro = metrics.f1_score(y_test, y_pred, average='micro')
print (f'Micro F-measure: {fmeasure_micro}')
fmeasure_macro = metrics.f1_score(y_test, y_pred, average='macro')
print (f'Macro F-measure: {fmeasure_macro}')
#输出报告
print(metrics.classification_report(y_test, y_pred))