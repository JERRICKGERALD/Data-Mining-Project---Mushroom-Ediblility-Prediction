#!/usr/bin/env python
# coding: utf-8

# In[34]:


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


# In[35]:


df=pd.read_csv(r'final.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df.head()


# In[36]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[37]:


features=["ring-type", "stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height","stem-width", "stem-height"]
prediction=['class']


# In[38]:


X=df[features]
y=df[prediction]


# #### Train Test Split

# In[39]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=10)


# ###  Random Forest Classifier

# In[40]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[41]:


print("Random Forest Classifier ACCURACY:",metrics.accuracy_score(y_test,y_pred))


# In[42]:


print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))


# In[44]:


cf_mat = confusion_matrix(y_test, y_pred)
cf_mat


# ### AUC - ROC Curve

# In[ ]:




