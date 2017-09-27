
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

import pandas as pd


# In[ ]:

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:

import sklearn
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[ ]:

import seaborn as sns


# In[ ]:

import pandas as pd

import sklearn
from sklearn import preprocessing

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

df = pd.read_table('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', sep =',',
                  names = ["buying","maint","doors","persons","lug_boot","safety","eval"])
df["eval"].value_counts()
import pandas as pd
car_counts = pd.DataFrame(df["eval"].value_counts())
car_counts

car_counts["Percentage"] = car_counts["eval"]/car_counts.sum()[0]
car_counts

plt.figure(figsize=(8,8))
plt.pie(car_counts["Percentage"],
       labels = ["Unacceptable","Acceptable","Good","Very good"])

le = preprocessing.LabelEncoder()
encoded_buying = le.fit(df["buying"])
encoded_buying.classes_
encoded_buying.transform(["high"])
encoded_buying.transform(["low"])
encoded_buying.transform(["med"])
encoded_buying.transform(["vhigh"])
encoded_buying.inverse_transform(1)


# In[ ]:

for i in range(0,4):
    print(i, ":", encoded_buying.inverse_transform(i))


# In[ ]:

df["e.buying"] = df["buying"].map(lambda x: encoded_buying.transform([x]))


# In[ ]:

print(df)


# In[ ]:

df.head()


# In[ ]:

df["e.buying"] = df["e.buying"].map(lambda x: x[0])
df.head()


# In[ ]:

df[pd.isnull(df).any(axis=1)]


# In[ ]:

encoded_maint = le.fit(df["maint"])
encoded_maint.classes_


# In[ ]:




# In[ ]:

df.head()


# In[ ]:




# In[ ]:

def encode_col (col_name):
    encodes = le.fit(df[col_name])
    new_col_name = "e."+col_name
    df[new_col_name] = df[col_name].map(lambda x:encodes.transform([x]))
    df[new_col_name] = df[new_col_name].map(lambda x:x[0])
    return



# In[ ]:

encode_col("maint")
encode_col("doors")
encode_col("persons")
encode_col("lug_boot")
encode_col("safety")
encode_col("eval")
df.head()


# In[ ]:

pd.DataFrame(df["eval"].value_counts())


# In[ ]:

pd.DataFrame(df["e.eval"].value_counts())


# In[ ]:

X = df[['e.buying','e.maint','e.doors','e.persons','e.lug_boot','e.safety']]
type(X)


# In[ ]:

X.shape


# In[ ]:

y = df[["e.eval"]]
type(y)


# In[ ]:

y.shape


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 5)


# In[ ]:

#X_train.shape
#X_test.shape
#y_train.shape
y_test.shape


# In[ ]:

clf_dt = tree.DecisionTreeClassifier(random_state=10)


# In[ ]:

clf_dt.fit(X_train, y_train)


# In[ ]:

y_pred_dt = clf_dt.predict(X_test)


# In[ ]:

type(y_pred_dt)


# In[ ]:

y_pred_dt.shape


# In[ ]:

y_pred_dt


# In[ ]:

print(metrics.accuracy_score(y_test, y_pred_dt))


# In[ ]:

correct_pred_dt = []
wrong_pred_dt = []


# In[ ]:

y_test2 = y_test.reset_index(drop=True)


# In[ ]:

y_test2 = y_test2.as_matrix()


# In[ ]:

for i in range(0,432):
    if y_test2[i] != y_pred_dt[i]:
        wrong_pred_dt.append(i)
    else:
        correct_pred_dt.append(i)


# In[ ]:

print("Correctly identified labels: ", len(correct_pred_dt))


# In[ ]:

print("Wrongly identified labels: ", len(wrong_pred_dt))


# In[ ]:

X_test.head()


# In[ ]:

y_test[0:5]


# In[ ]:

y_pred_dt[0:5]


# In[ ]:

def dt_probs (index_num):
    X_param = X_test.loc[index_num]
    X_param = X_param.to_frame()
    X_param = X_param.transpose()
    temp_pred = clf_dt.predict_proba(X_param)
    temp_pred_1 = temp_pred[0]
    y_actual = y_test[index_num]
    y_range = ["Unacceptable","Acceptable","Good","Very Good"]
    print("For index num: ",index_num)
    print("Features entered: ",X_param)
    print("y actual")
    print(y_actual, "(",y_range[y_actual],")")
    print("predicted probabilities:")
    
    for i in range(0,4):
        print(y_range[i],":", temp_pred_1[i])
    return

#dt_probs(805)


# In[ ]:

dt_probs(805)


# In[ ]:

dt_probs(50)


# In[ ]:

for i in range(0, 432):
    if y_pred_dt[i] != y_test2[i]:
        print (i)


# In[ ]:

X_test.head()


# In[ ]:

print(y_test)
y_test3 = y_test.toframe()
#y_test3 = y_test3.reset_index()


# In[ ]:

y_test3 = y_test3.reset_index()


# In[ ]:

y_test3.head()


# In[ ]:

y_test3.ix[19]


# In[ ]:

dt_probs(1130)


# In[ ]:

y_test3.ix[41]


# In[ ]:

x_test.ix[1130]


# In[ ]:

dt_probs(1130)


# In[ ]:



