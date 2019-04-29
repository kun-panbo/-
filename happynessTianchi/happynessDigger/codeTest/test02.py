# -*- coding: utf-8 -*-
"""

Created on Sun Apr 28 14:28:11 2019

@author: hepanbo

E-mail: panbohero@126.com

day day up up!

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import lightgbm as lgb
train = pd.read_csv(r"../resource/happiness_train_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
test = pd.read_csv(r"../resource/happiness_test_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')

#主要是用年龄替换掉了两个属性，降维
train['survey_time'] = train['survey_time'].dt.year
test['survey_time'] = test['survey_time'].dt.year
#print(test.head())
train['Age']=train['survey_time']-train['birth']
test['Age']=test['survey_time']-test['birth']
#print(train['Age'])
del_list=['survey_time','birth']

features=['Age','inc_ability','gender','status_peer','work_exper','family_status','health','equity','class','health_problem','family_m','house','depression','learn','relax','edu']
target = train['happiness']
train = train[features]
test = test[features]
feature_importance_df = pd.DataFrame()
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

# 设置参数
params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves':31 ,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 2,
            'header': True
            }


kfolds = KFold(n_splits=5,shuffle=True,random_state=15)
for fold_n,(trn_index,val_index) in enumerate(kfolds.split(train,target)):
    print("fold_n {}".format(fold_n))
    trn_data = lgb.Dataset(train.iloc[trn_index],label=target.iloc[trn_index])
    val_data = lgb.Dataset(train.iloc[val_index],label=target.iloc[val_index])
    num_round=10000
    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
    oof[val_index] = clf.predict(train.iloc[val_index], num_iteration=clf.best_iteration)
    predictions += clf.predict(test,num_iteration=clf.best_iteration)/5
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_n + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

test = pd.read_csv(r"../resource/happiness_test_abbr.csv",parse_dates=['survey_time'],encoding='latin-1')
submision_lgb1  = pd.DataFrame({"id":test['id'].values})
submision_lgb1["happiness"]=predictions
submision_lgb1.to_csv("submision_lgb02.csv",index=False)