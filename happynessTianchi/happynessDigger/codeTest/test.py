# -*- coding: utf-8 -*-
"""

Created on Tue Apr  9 16:44:41 2019

@author: hepanbo

E-mail: panbohero@126.com

day day up up!

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#读取数据
#train_data = pd.read_csv(r"../resource/happiness_train_abbr.csv")
train = pd.read_csv(r"../resource/happiness_train_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
test = pd.read_csv(r"../resource/happiness_test_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
#print(train_data.info())
#看幸福的set有哪几种
#happy=set(train_data['happiness'])
#修改-8  无回答  改成6 第六类：无法回答

# =============================================================================
# train_data['happiness'] = train_data['happiness'].replace( -8, 6 )
# happy=set(train_data['happiness'])
# 
# =============================================================================


#研究happiness的分布
# =============================================================================
# print(train_data['happiness'].describe())
# 
# sns.set_style("whitegrid")
# 
# sns.distplot(train_data['happiness'])
# =============================================================================


#df = train_data['happiness'].fillna(0)
# =============================================================================
# df=train_data['family_income'].fillna(0)
# print(df.describe()) 
# print(train_data.info())
# =============================================================================
# =============================================================================
# print('shape is ',train_data.shape)
# print('describe is ',train_data.describe())
# =============================================================================


#print(train_data.isnull().sum())
#print(train_data['work_status'].describe())

# =============================================================================
# train = train_data.loc[train_data['happiness']!=-8]
# 
# f,ax = plt.subplots(1,2,figsize=(18,8))
# train['happiness'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
# ax[0].set_title('happiness')
# ax[0].set_ylabel('')
# sns.countplot('happiness',data=train, ax=ax[1])
# ax[1].set_title('happiness')
# plt.show()
# =============================================================================
#print(train['survey_time'][:5])


#主要是用年龄替换掉了两个属性，降维
train['survey_time'] = train['survey_time'].dt.year
test['survey_time'] = test['survey_time'].dt.year
#print(test.head())
train['Age']=train['survey_time']-train['birth']
test['Age']=test['survey_time']-test['birth']
#print(train['Age'])
del_list=['survey_time','birth']

# =============================================================================
# sns.heatmap(train[['happiness','Age','inc_ability','gender','status_peer','family_status','health','equity','class','work_exper','health_problem','family_m','house','depression','learn','relax','edu']].corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
# fig=plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()
# =============================================================================

features=['Age','inc_ability','gender','status_peer','work_exper','family_status','health','equity','class','health_problem','family_m','house','depression','learn','relax','edu']
target = train['happiness']
train = train[features]
test = test[features]
feature_importance_df = pd.DataFrame()
oof = np.zeros(len(train))
predictions = np.zeros(len(test))


params = {'num_leaves': 9,
         'min_data_in_leaf': 40,
         'objective': 'regression',
         'max_depth': 16,
         'learning_rate': 0.01,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,   # 每次迭代时用的数据比例
         'feature_fraction': 0.8,# 每次迭代中随机选择80％的参数来建树
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'rmse',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}

kfolds = KFold(n_splits=5,shuffle=True,random_state=15)



clf = SVC()
clf.fit(train,target)
predictions=clf.predict(test)
test = pd.read_csv(r"../resource/happiness_test_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
submision_lgb1  = pd.DataFrame({"id":test['id'].values})
submision_lgb1["happiness"]=predictions
submision_lgb1.to_csv("submision_svm1.csv",index=False)

# =============================================================================
# 
# kfolds = KFold(n_splits=5,shuffle=True,random_state=15)
# for fold_n,(trn_index,val_index) in enumerate(kfolds.split(train,target)):
#     print("fold_n {}".format(fold_n))
#     trn_data = lgb.Dataset(train.iloc[trn_index],label=target.iloc[trn_index])
#     val_data = lgb.Dataset(train.iloc[val_index],label=target.iloc[val_index])
#     num_round=10000
#     clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
#     oof[val_index] = clf.predict(train.iloc[val_index], num_iteration=clf.best_iteration)
#     predictions += clf.predict(test,num_iteration=clf.best_iteration)/5
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["feature"] = features
#     fold_importance_df["importance"] = clf.feature_importance()
#     fold_importance_df["fold"] = fold_n + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
# 
# 
# test = pd.read_csv(r"../resource/happiness_test_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
# submision_lgb1  = pd.DataFrame({"id":test['id'].values})
# submision_lgb1["happiness"]=predictions
# submision_lgb1.to_csv("submision_lgb1.csv",index=False)
# #['happiness','Age','inc_ability','gender','status_peer','family_status','health','equity','class','work_exper','health_problem','family_m','house','depression','learn','relax','edu']
# =============================================================================
