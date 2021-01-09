# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:45:04 2021

@author: Asus
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
X, y= make_blobs(n_samples=2000, n_features=3)

data= pd.DataFrame(X ,columns = ["column_1","column_2", "column_3"])
#also we have label column (y)
print(data.head(10))

print(data.info())
#we have float Dtype

print(data.isna().sum())
#we have not any missing value

print(data.describe())
#we can see some statistical information about data




#how many values for each labels
y= pd.DataFrame(y, columns=["label"])
print(y["label"].value_counts())



X= pd.DataFrame(X ,columns = ["column_1","column_2", "column_3"])
data_concat = pd.concat([X, y], axis=1)
plt.figure(figsize=(12,8))
sns.heatmap(data_concat.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()

from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier


X_train, X_test, y_train, y_test = train_test_split(data, y ,test_size=0.3, random_state=0)


clf = DecisionTreeClassifier()
#we have to define max_depth to prevent overfitting
clf.fit(X_train,y_train)
print("Train Accuracy of clf:",clf.score(X_train,y_train))
print("Test Accuracy of clf",clf.score(X_test,y_test))



xgb = XGBClassifier()
xgb.fit(X_train,y_train)
print("Train Accuracy of xgb:",xgb.score(X_train,y_train))
print("Test Accuracy of xgb:",xgb.score(X_test,y_test))

#%%
from sklearn.model_selection import GridSearchCV

#GridSearch on Xgboost Classifier
param_dict = {
    'max_depth':range(2,3,4),
    'min_child_weight':range(1,2,6),
    'learning_rate': [0.00001,0.001,0.01,0.1],
    'n_estimators': [10,50,100]}

xgb_ = GridSearchCV(xgb,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(xgb_.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(xgb_.best_score_))
print("Train Score {:.6f}".format(xgb_.score(X_train,y_train)))
print("Test Score {:.6f}".format(xgb_.score(X_test,y_test)))
#%%
print("****************************")
#GridSearch on Decision Tree Classifier
param_dict = {
    'max_depth':range(3,5),
    'criterion': ["gini", "entropy"]}

clf_ = GridSearchCV(clf,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(clf_.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(clf_.best_score_))
print("Train Score {:.6f}".format(clf_.score(X_train,y_train)))
print("Test Score {:.6f}".format(clf_.score(X_test,y_test)))
#%%
#Evaluate your result on both train and test set. 
#Analyse if there is any underfitting or overfitting problem. Make your comments.



#firstly I have to say our data is very small and not complicated especially our models.
#we can realize that our models have dramatically overfitting problems.
#in my opinion for this situation gridSearch is not necessary also.
#if I compare these two models of course Xgboost more complicated than other. But for this dataset
#Xgboost couldnt shown its ability and advantages.
print("Best Score with XGboost Classifier {:.6f}".format(xgb_.best_score_))