#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from sklearn import metrics
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[58]:


df=pd.read_csv(r'final.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df.head()


# In[59]:


df.shape


# In[60]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[61]:


features=["ring-type", "stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height","stem-width", "stem-height"]
prediction=['class']


# In[62]:


X=df[features]
y=df[prediction]


# #### Train Test Split

# In[63]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)


# ###  Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[65]:


print("Random Forest Classifier ACCURACY:",metrics.accuracy_score(y_test,y_pred))


# In[66]:


print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))


# In[75]:


model = RandomForestClassifier(max_depth=2, n_estimators=30,min_samples_split=3, max_leaf_nodes=5,random_state=22)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Training Accuracy : ',metrics.accuracy_score(y_train,model.predict(X_train))*100)
print('Validation Accuracy : ',metrics.accuracy_score(y_test,model.predict(X_test))*100)


# In[76]:


print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))


# In[ ]:





# In[77]:


cf_mat = confusion_matrix(y_test, y_pred)
print(cf_mat)
sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, 
            fmt='.2%', cmap='Blues')


# In[78]:


conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[ ]:





# In[79]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

plt.legend()


# ### AUC - ROC Curve

# In[80]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

#add legend
plt.legend()


# In[ ]:




