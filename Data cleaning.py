#!/usr/bin/env python
# coding: utf-8

# ### Mushroom Edibility Prediction

# In[17]:


import pandas as pd
import numpy as np


# #### Importing data

# In[18]:


data = pd.read_csv(r"C:\Users\Jerrick\Downloads\secondary_data.csv\secondary_data.csv",sep=';')
data.head()


# In[19]:


data['class'].value_counts()


# In[20]:


missing_values_count = data.isnull().sum()


# In[21]:


missing_values_count


# #### Dropping missing values

# In[22]:


data=data.drop(['cap-surface','gill-spacing','stem-root','stem-surface','veil-type','veil-color','spore-print-color'],axis=1)
data.head()


# #### Checking missing values 

# In[25]:


data= data.fillna(0)
missing=data.isnull().sum()
missing


# In[26]:


data.head()


# ### Creating Final Csv

# In[27]:


data.to_csv('final.csv')


# In[29]:


df=pd.read_csv('final.csv')
df.shape

