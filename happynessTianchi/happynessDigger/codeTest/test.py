# -*- coding: utf-8 -*-
"""

Created on Tue Apr  9 16:44:41 2019

@author: hepanbo

E-mail: panbohero@126.com

day day up up!

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#读取数据
train_data = pd.read_csv(r"../resource/happiness_train_abbr.csv")
#print(train_data.info())
#看幸福的set有哪几种
#happy=set(train_data['happiness'])
#修改-8  无回答  改成6 第六类：无法回答

train_data['happiness'] = train_data['happiness'].replace( -8, 6 )
happy=set(train_data['happiness'])



#研究happiness的分布
# =============================================================================
# print(train_data['happiness'].describe())
# 
# sns.set_style("whitegrid")
# 
# sns.distplot(train_data['happiness'])
# =============================================================================


df = train_data['happiness'].fillna(0)
 
print(df.describe())

