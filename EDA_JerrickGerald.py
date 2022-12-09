#!/usr/bin/env python
# coding: utf-8

# ## Mushroom Edibility prediction
# 
# 
# ### Import Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r'final.csv')
df=df.drop(['Unnamed: 0'],axis=1)
df.head()


# In[14]:


df.describe()


# ### CLASS DISTRIBUTION
# 
# #### P - Poisonous, E - Edible

# In[3]:


df['class'].value_counts()


# In[4]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[5]:


count_classes = pd.value_counts(df['class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)


# ### So, from the distribution we can see that  we have higher number of poisonous class when compare to edible mushroom class. And no need of balancing technique is required for this dataset.

# ### Distribution of Stem - Color

# In[6]:


df.head()


# In[7]:


count_classes = pd.value_counts(df['stem-color'], sort = True)
count_classes.plot(kind = 'bar', rot=0,color='tan')


# In[ ]:





# 

# In[9]:


plt.figure(figsize=(16, 10))
comat=df.corr()
k=15
col=comat.nlargest(k,'class')['class'].index
cm=np.corrcoef(df[col].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,annot=True,cbar=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=col.values, xticklabels=col.values)
plt.show()


# In[12]:


sns.boxplot(y='stem-color',x='class',data=df)


# In[ ]:




