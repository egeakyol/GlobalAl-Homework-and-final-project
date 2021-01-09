# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 02:41:53 2021

@author: Asus
"""

#import libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split
from scipy import stats
from sklearn.datasets import load_boston

#Import Boston Dataset from sklearn dataset class
X, y =  load_boston(return_X_y = True)

#%%
#Explore and analyse raw data
df_boston = pd.DataFrame(X ,columns = load_boston().feature_names)
print(df_boston.head(10))
#default values is 5 for head method.

print(df_boston.columns)
#look at columns names

print(df_boston.info())
# we can check samples types and counts

print(df_boston.describe())
#we can see that some statistical information about data
#df_boston["column_name"] also we can use for specific information about a column

print(df_boston.isna().sum())
#df_boston.isna() turn us "True" labels if there are any missing value in data otherwise "False"
#we checked the missing 
#%%

#Do preprocessing for regression.
plt.figure(figsize=(18,10))
sns.heatmap(df_boston.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()

#regressions doesnt need to any scaling process at the sametime regressions affects outliers
z = np.abs(stats.zscore(df_boston))

outliers = list(set(np.where(z > 3)[0]))
new_df = df_boston.drop(outliers,axis = 0).reset_index(drop = False)

print("outlier samples :{}".format(z))
print("number of outliers: {}".format(len(outliers)))




#we created new data which is not comprises any outliers
X_new = new_df.drop('index', axis = 1)
y_new = y[list(new_df["index"])]
print(len(y_new))
print(len(X_new))

#%%
#Split your dataset into train and test test (0.7 for train and 0.3 for test).
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new ,test_size=0.3, random_state=0)
print(X_train.shape)



#%%
#Try Ridge and Lasso Regression models with at least 5 different alpha value for each.
from sklearn.linear_model import  LinearRegression, Ridge, Lasso
regression = LinearRegression()
model = regression.fit(X_train, y_train)
print("Linear Regression Train: ", regression.score(X_train, y_train))
print("Linear Regression  Test: ", regression.score(X_test, y_test))
print('intercept:', model.intercept_)

importance = model.coef_
for i in range(len(importance)):
    print("Feature", df_boston.columns[i], "Score:", importance[i])
#we can remove a few columns which has low coef_ scores for better score.
print('*************************')


alpha_values= [1, 0.1, 0.01, 0.001, 0.0001]

#Regularized Ridge Regression
print("********************Ridge Regression ********************")

for i in alpha_values:
    ridge_model = Ridge(alpha = i)
    ridge_model.fit(X_train, y_train)
    print("Ridge Train: ", ridge_model.score(X_train, y_train))
    print("Ridge Test: ", ridge_model.score(X_test, y_test))
    print('**************')


#Regularized Lasso Regression
print("*********************Lasso Regression *********************")
for i in alpha_values:
    lasso_model = Lasso(alpha = i)
    lasso_model.fit(X_train, y_train)
    print("Lasso Train: ", lasso_model.score(X_train, y_train))
    print("Lasso Test: ", lasso_model.score(X_test, y_test))
    print('**************')





#%%
#Evaluate the results of all models and choose the best performing model.

#best alpha value is 0.01 for lasso regresion models 
#best alpha value is 0.1 for ridge regresion models 


print("best performing model is;")
ridge_model = Ridge(alpha = 0.1)
ridge_model.fit(X_train, y_train)
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))







