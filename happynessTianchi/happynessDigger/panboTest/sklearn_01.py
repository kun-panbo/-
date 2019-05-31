# -*- coding: utf-8 -*-
"""

Created on Mon Apr 22 15:22:58 2019

@author: hepanbo

E-mail: panbohero@126.com

day day up up!

"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits

from sklearn import preprocessing

from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC


from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np



iris = datasets.load_iris()
X,y = iris.data, iris.target
#clf=SVC()
#clf.fit(X,y)
#joblib.dump(clf,'clf.pkl')


clf3 = joblib.load('clf.pkl')
print(clf3.predict(X))


# =============================================================================
# digits = load_digits()
# X=digits.data
# y=digits.target
# 
# train_size, train_loss, test_loss = learning_curve(
#         SVC(gamma=0.1), X, y,cv=10, scoring ='neg_mean_squared_error',
#         train_sizes=[0.1,0.25,0.5,0.75,1])
# 
# train_loss_mean = -np.mean(train_loss,axis=1)
# test_loss_mean= -np.mean(test_loss,axis=1)
# plt.figure()
# 
# plt.plot(train_size,train_loss_mean,'o-',color='r',label='Training')
# plt.plot(train_size,test_loss_mean,'o-',color='g',label='Cross-validation')
# 
# plt.show()
# 
# 
# =============================================================================
# =============================================================================
# iris = datasets.load_iris()
# iris_X = iris.data 
# iris_y = iris.target
# 
# k_range = range(10,20)
# k_score=[]
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=5)
#     scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
#     k_score.append(scores.mean())
# 
# plt.figure()
# plt.plot(k_range,k_score)
# plt.show()
# =============================================================================
# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y,test_size =0.3)
# 
# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn,X,y,cv=5,scoring='accuracy')
# print(scores.mean())
# =============================================================================


# =============================================================================
# knn.fit(X_train, y_train)
# 
# print(knn.predict(X_test))
# print(y_test)
# =============================================================================

# =============================================================================
# 
# X, y = datasets.make_regression(n_samples=100, n_features =1 , n_targets=1, noise=1)
# plt.figure()
# plt.scatter(X,y)
# plt.show()
# =============================================================================
# =============================================================================
# 
# load_data = datasets.load_boston()
# data_X = load_data.data
# data_y = load_data.target
# 
# print(data_X.shape)
# 
# 
# model = LinearRegression()
# model.fit(data_X, data_y)
# model.predict(data_X[:4,:])
# 
# =============================================================================
# =============================================================================
# a=np.array([[10,2.7,3.6],
#             [-100,5,-2],
#             [120,20,40]],dtype=np.float64)
# print(a)
# 
# print(preprocessing.scale(a))   #将值得相差度减小
# 
# =============================================================================

# =============================================================================
# plt.figure
# X,y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state = 22, n_clusters_per_class=1, scale =100)
# plt.scatter(X[:,0],X[:,1], c=y)
# #plt.scatter(X,y)
# plt.show()
# 
# X=preprocessing.minmax_scale(X)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
# clf = SVC()  #支持向量机
# clf.fit(X_train, y_train)
# print(clf.score(X_test,y_test))
# =============================================================================
