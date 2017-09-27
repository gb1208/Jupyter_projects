
# coding: utf-8

# In[ ]:

import pandas as pd


# In[ ]:

import sklearn


# In[ ]:

print (sklearn.__version__)


# In[ ]:

import numpy as np


# In[ ]:

import matplotlib.pyplot as plt


# In[ ]:

#matplotlib.style.use('ggplot')
#%matplotlib inline


# In[ ]:

from sklearn.model_selection import train_test_split


# In[ ]:

from sklearn import linear_model


# In[ ]:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[ ]:

import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", delimiter = r"\s+",
                    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"])
#df.shape
#df[pd.isnull(df).any(axis=1)]
X = df[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]]
y = df["MEDV"]
type(X)
X.shape
type(y)
y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 5)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.intercept_


# In[ ]:

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.intercept_
reg.coef_
reg.predict([[0.03237, 0.0, 2.18, 0, 0.458, 6.998, 45.8,6.0622, 3, 222.0, 18.7, 394.63, 2.94]])


# In[ ]:

X_test.head()


# In[ ]:

y_test[0:5]


# In[ ]:

reg.predict([[0.38214, 0.0, 6.20, 0, 0.504, 8.040, 86.5,3.2157, 8, 307.0, 17.4, 387.38, 3.13]])


# In[ ]:

print(X_test)


# In[ ]:

y_pred = reg.predict(X_test)
print(y_pred)
type(y_pred)


# In[ ]:

len(y_pred)


# In[ ]:

y_pred[0:5]


# In[ ]:

type(y_test)


# In[1]:

import pandas
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", delimiter = r"\s+",
                    names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"])
#df.shape
#df[pd.isnull(df).any(axis=1)]
X = df[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]]
y = df["MEDV"]
type(X)
X.shape
type(y)
y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 5)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.intercept_
y_pred = reg.predict(X_test)
y_pred[0:5]
y_test_m = y_test.as_matrix()
print(y_test_m)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
plt.plot(y_test_m,ms=50, alpha=1)
plt.plot(y_pred,ms=50, alpha=1)
legend_list = ['y_test_m','y_pred']
plt.legend(legend_list,loc=4, fontsize='25')
mean_squared(y_test,y_pred)


# In[ ]:

type(y_test_m)


# In[9]:

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(15,10))
plt.plot(y_test_m,ms=50, alpha=1)
plt.plot(y_pred,ms=50, alpha=1)
legend_list = ['y_test_m','y_pred']
plt.legend(legend_list,loc=4, fontsize='25')


# In[6]:

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mean_squared_error(y_test,y_pred)


# In[7]:

r2_score(y_test, y_pred)


# In[ ]:



