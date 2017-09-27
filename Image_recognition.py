
# coding: utf-8

# In[1]:

import numpy as np
import sklearn
import pandas as pd


# In[31]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# In[2]:

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data', delimiter=r"\s+",
                header=None)


# In[3]:

df.head()


# In[4]:

df.shape


# In[5]:

X = pd.DataFrame(df)


# In[6]:

X = X.drop([256, 257, 258, 259, 260, 261, 262, 262, 263, 264, 265], axis =1)


# In[7]:

type(X)


# In[8]:

X.shape


# In[9]:

label_df = pd.DataFrame(df.iloc[:,[256, 257, 258, 259, 260, 261, 262, 262, 263, 264, 265]])


# In[10]:

label_df.shape


# In[11]:

label_df.head()


# In[13]:

label_df.rename(columns={256:0, 257:1, 258:2, 259:3, 260:4, 261:5, 262:6, 263:7, 264:8, 265:9}, inplace = True)


# In[14]:

label_df.head()


# In[15]:

label_df["y"] = label_df.apply(lambda x: label_df.columns[x.argmax()], axis = 1label_df)


# In[16]:

label_df.head()


# In[17]:

label_df.tail()


# In[19]:

y = label_df["y"]


# In[20]:

type(y)


# In[23]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 5)


# In[24]:

X_train.shape


# In[25]:

X_test.shape


# In[26]:

y_train.shape


# In[27]:

y_test.shape


# In[ ]:




# In[32]:

clf_knn = KNeighborsClassifier()


# In[36]:

k_range = range(1, 6)


# In[37]:

knn_weight_options = ["uniform", "distance"]


# In[38]:

knn_algorithm_options = ["ball_tree", "kd_tree", "brute"]


# In[39]:

knn_params = {"n_neighbors":k_range, "weights":knn_weight_options, "algorithm":knn_algorithm_options}


# In[43]:

knn_grid = GridSearchCV(clf_knn, knn_params, cv = 10, scoring = "accuracy")


# In[44]:

knn_grid.fit(X, y)


# In[45]:

knn_grid.grid_scores_


# In[46]:

knn_grid.best_params_


# In[47]:

knn_grid.best_score_


# In[ ]:




# In[53]:

clf_dt = tree.DecisionTreeClassifier(random_state = 10)


# In[54]:

dt_splitter = ["best","random"]


# In[55]:

dt_criterion = ["gini", "entropy"]


# In[56]:

dt_params = {"splitter":dt_splitter, "criterion":dt_criterion}


# In[57]:

dt_grid = GridSearchCV(clf_dt, dt_params, cv = 10, scoring = "accuracy")


# In[58]:

dt_grid.fit(X, y)


# In[59]:

dt_grid.grid_scores_


# In[60]:

dt_grid.best_params_


# In[61]:

dt_grid.best_score_


# In[62]:

dt_grid.best_estimator_


# In[ ]:




# In[63]:

clf_final = KNeighborsClassifier(algorithm = "brute", leaf_size = 30, metric="minkowski",
                               metric_params = None, n_jobs = 1, n_neighbors = 4, p =2,
                               weights = "distance")


# In[64]:

clf_final.fit(X_train, y_train)


# In[65]:

y_pred = clf_final.predict(X_test)


# In[66]:

print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:




# In[69]:

cm = confusion_matrix(y_test, y_pred)


# In[70]:

cm


# In[71]:

labels  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df_cm = pd.DataFrame(cm, index = [i for i in labels], 
                     columns = [i for i in labels])

plt.figure(figsize = (10, 10))
sns.heatmap(df_cm, annot = True)
plt.xlabel("predicted", fontsize = 20)
plt.ylabel("actual", fontsize = 20)


# In[ ]:




# In[72]:

scr_clf_knn = precision_recall_fscore_support(y_test, y_pred, average = "weighted")


# In[74]:

print("classifier's precision: "+str(scr_clf_knn[0]))
print("classifier's recal: "+str(scr_clf_knn[1]))
print("classifier's fbeta_score: "+str(scr_clf_knn[2]))


# In[ ]:




# In[75]:

X_test.head()


# In[76]:

y_test[:5]


# In[77]:

type(y_test)


# In[78]:

y_test_df = y_test.to_frame()


# In[79]:

y_test_df.head()


# In[80]:

y_test_df["y_pred"] = pd.Series(y_pred, index = y_test_df.index)


# In[81]:

y_test_df.head()


# In[82]:

y_test_df.index[y_test_df.y != y_test_df.y_pred]


# In[83]:

wrong_list = [1519, 149, 1153, 1158, 1562, 568]
y_test_df.ix[wrong_list]


# In[ ]:




# In[86]:

def make_image(index_num):
    one_row = X.ix[index_num]
    one_values = one_row.values
    
    i = 16
    j = 0
    img = np.array(one_values[:16])
    while i <= len(one_values):
        temp_array = np.array(one_values[j:i])
        img = np.vstack((img, temp_array))
        j = 1
        i += 16
        
    plt.imshow(img, cmap = plt.cm.gray_r, interpolation = "nearest")
    plt.show()
    
    print("*-*-*-*-*-*-*-*-*-*")
        
    print(y_test_df.ix[index_num])
    return


# In[88]:

make_image(1519)


# In[89]:

make_image(149)


# In[ ]:

make_image(1153)


# In[ ]:

make_image(1158)


# In[ ]:

make_image(1562)


# In[ ]:

make_image(568)


# In[ ]:

make_image(599)


# In[ ]:

make_image(977)


# In[ ]:

make_image(362)


# In[ ]:



