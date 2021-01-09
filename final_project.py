# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 20:04:18 2021

@author: Asus
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix



data = pd.read_csv("winequality.csv")

print(data.head(10))

print(data.info())


print(data.isna().sum())
#we have not any missing value. Good !

print(data.describe())

print(data.columns)


#visualization
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x="quality", data=data, palette="bwr")
plt.xticks(np.arange(6))
plt.xlabel("Default Payent" , color="red", alpha=0.7, size=22)
plt.show()

print(data["quality"].value_counts())
#I think that we can removed some labels for more accuracy

plt.figure(figsize=(12,6))
sns.set_style('whitegrid') 
sns.distplot(data['alcohol'], kde = True, color ='red', bins = 10) 
plt.show()
#%% Feature extraction

#for first question scaling process depends on our models. I didnt use these model because of that ı didnt use
#scaling at the same time future make more precision our models.
data_removed = data[(data["quality"] !=8) & (data["quality"] !=3 ) & (data["quality"] !=4 )]


#duplicated rows
print("duplicated samples len: ",data_removed.duplicated().sum())
# we have a lot of duplicated samples but ı dont removed that samples because ı dont want to lose our limited samples.


#most related columns with quality
plt.figure(figsize=(12,8))
plt.scatter(x="volatile acidity", y="alcohol", data = data_removed[data_removed["quality"]==5], color ="red")
plt.scatter(x="volatile acidity", y="alcohol", data = data_removed[data_removed["quality"]==6], color ="green")
plt.scatter(x="volatile acidity", y="alcohol", data = data_removed[data_removed["quality"]==7], color ="blue")
plt.xlabel("volatile acidity")
plt.ylabel("alcohol")
plt.show()

#in my opinion I make feature exctruction but this situation not necessary.  
#%% Build Models

X = data_removed.iloc[:,:-1]
y = data_removed.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
print("***********Decision Tree Classifier************")
print("Train Accuracy :",round(clf.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,clf_pred),3))



xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
print("*************Xgboost Classifier************")
print("Train Accuracy of xgb:",round(xgb.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,xgb_pred),3))


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("*************Random Forest Classifier************")
print("Train Accuracy of Random Forest C:",round(rfc.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(rfc_pred,xgb_pred),3))
#%%   Regularization
from sklearn.model_selection import GridSearchCV

#GridSearch on Decision Tree Classifier
param_dict = {
    'max_depth':range(3,5),
    'criterion': ["gini", "entropy"]}

clf_ = GridSearchCV(clf,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)
print("Mean of the cv scores is {:.5f}".format(clf_.best_score_))
print("Best Parameters{}".format(clf_.best_params_))
print("*************")


#GridSearch on Xgboost Classifier
param_dict = {
    'max_depth':range(3,4),
    'min_child_weight':range(1,2,6),
    'learning_rate': [0.00001,0.001,0.01,0.1],
    'n_estimators': [10,30,50,80,100]}

xgb_ = GridSearchCV(xgb,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)
print("Mean of the cv scores is {:.5f}".format(xgb_.best_score_))
print("Best Parameters{}".format(xgb_.best_params_))
print("*************")


#GridSearch on Random Forest Classifier
param_dict = {
    'n_estimators': [20,30,40,50],
    'min_samples_split': [2,3,4]}

rfc_ = GridSearchCV(rfc,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train) 
print("Mean of the cv scores is {:.5f}".format(rfc_.best_score_))
print("Best Parameters{}".format(rfc_.best_params_))
print("*************")
#%%
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#in this part we run the models with best parameters which getting with GridSearch method. 
#After than we plot confusion matrixs. and priting the console its different metrics which are recall precision 
print("***********Decision Tree Classifier************")
clf = DecisionTreeClassifier(criterion ="gini", max_depth = 3)
clf.fit(X_train,y_train)
clf_pred = clf.predict(X_test)
print("Train Accuracy :",round(clf.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,clf_pred),3))

plot_confusion_matrix(clf ,X_test, y_test)  
plt.title("Decision Tree Classifier")
plt.show()

print("Precision of Decision Tree Classifier :{}".format(precision_score(y_test, clf_pred, average='macro')))
print("Recall of Decision Tree Classifier : {}".format(recall_score(y_test, clf_pred, average='macro')))
print("f1-score of Decision Tree Classifier :{}".format(f1_score(y_test, clf_pred, average="macro")))
print("Accuracy of Decision Tree Classifier :{}".format(accuracy_score(y_test, clf_pred)),"\n")
print("****************************")


print("*************Xgboost Classifier************")
xgb = XGBClassifier(learning_rate = 0.1, max_depth= 3, min_child_weight= 1, n_estimators=100)
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)
print("Train Accuracy of xgb:",round(xgb.score(X_train,y_train),3))
print("Test Accuracy ",round(metrics.accuracy_score(y_test,xgb_pred),3))


plot_confusion_matrix(xgb ,X_test, y_test)  
plt.title("XGboost Classifier")
plt.show()

print("Precision of XGBClassifier : {}".format(precision_score(y_test, xgb_pred, average='macro')))
print("Recall of XGBClassifier :{}".format(recall_score(y_test, xgb_pred, average='macro')))
print("f1-score of XGBClassifier:{}".format(f1_score(y_test, xgb_pred, average="macro")))
print("Accuracy  of XGBClassifier : {}".format(accuracy_score(y_test, xgb_pred)),"\n")
print("****************************")


print("*************Random Forest Classifier************")
rfc= RandomForestClassifier(min_samples_split= 2, n_estimators = 20)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("Train Accuracy of Random Forest C.:",round(rfc.score(X_train,y_train),3))
print("Test Accuracy",round(metrics.accuracy_score(y_test,rfc_pred),3))

plot_confusion_matrix(rfc ,X_test, y_test)  
plt.title("RandomForestClassifier")
plt.show()

print("Precision of Random Forest C. : {}".format(precision_score(y_test, rfc_pred, average='macro')))
print("Recall  of Random Forest C. : {}".format(recall_score(y_test, rfc_pred, average='macro')))
print("f1-score of Random Forest C.:{}".format(f1_score(y_test, rfc_pred, average="macro")))
print("Accuracy  of Random Forest C. : {}".format(accuracy_score(y_test, rfc_pred)))
print("****************************")
#%% comments and final decision.

#the best performing model is Random Forest Classsifier for my project. 
#For train dataset there is a bit overfitting signal but ı think its acceptable beacues our dataset is so small
#these tree algorithm creating to handle huge and complicated datesets. 
#finally our main topic must be precision on this project because of that we have there labels(we reduced to there).


("Best model is Random forest and our main metric is precision")
print("Accuracy  of Random Forest C. : {}".format(accuracy_score(y_test, rfc_pred)))
print("Precision of Random Forest C. : {}".format(precision_score(y_test, rfc_pred, average='macro')))