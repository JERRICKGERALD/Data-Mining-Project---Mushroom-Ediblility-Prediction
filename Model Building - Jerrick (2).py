#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df=pd.read_csv(r'final.csv')
df=df.drop(['Unnamed: 0','has-ring'],axis=1)
df.head()


# In[3]:


df.shape


# In[4]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
columns=['cap-shape','cap-color','does-bruise-or-bleed','gill-attachment','gill-color','stem-color','ring-type','habitat','season','class']
for column in columns:    
    df[column] = labelencoder.fit_transform(df[column])


# In[5]:


df.head()


# In[6]:


features=["stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height","stem-width", "stem-height"]
prediction=['class']


# In[7]:


X=df[features]
y=df[prediction]


# #### Train Test Split

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)


# ###  Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[ ]:


print("Random Forest Classifier ACCURACY:",metrics.accuracy_score(y_test,y_pred))


# In[ ]:


print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))


# In[ ]:


model = RandomForestClassifier(max_depth=2, n_estimators=30,min_samples_split=3, max_leaf_nodes=5,random_state=22)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Training Accuracy : ',metrics.accuracy_score(y_train,model.predict(X_train))*100)
print('Validation Accuracy : ',metrics.accuracy_score(y_test,model.predict(X_test))*100)


# In[ ]:


print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))


# In[ ]:


cf_mat = confusion_matrix(y_test, y_pred)
print(cf_mat)
sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, 
            fmt='.2%', cmap='Blues')


# In[ ]:


conf_matrix = confusion_matrix(y_test,y_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# ### AUC - ROC Curve

# In[ ]:


import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 


# In[ ]:



from sklearn .metrics import roc_auc_score

auc = np.round(roc_auc_score(y_test, y_pred), 3)

print("AUC SCORE",format(auc))


# In[ ]:




