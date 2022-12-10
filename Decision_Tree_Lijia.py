# Decision Tree
# By: Lijia Ren



#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#%%
df = pd.read_csv('final.csv')
df.info()
print(df.head())

#%%


df["class"]=df["class"].astype('category')
df["ring-type"]=df["ring-type"].astype('category')
df["cap-shape"]=df["cap-shape"].astype('category')
df["cap-color"]= df["cap-color"].astype('category')
df["does-bruise-or-bleed"]= df["does-bruise-or-bleed"].astype('category')
df["gill-attachment"]= df["gill-attachment"].astype('category')
df["gill-color"]= df["gill-color"].astype('category')
df["stem-color"]= df["stem-color"].astype('category')
df["has-ring"]= df["has-ring"].astype('category')
df["habitat"]= df["habitat"].astype('category')
df["season"]= df["season"].astype('category')

#%%
df.info()
#%%
#X:
features=["ring-type","stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color",  "stem-height"]
X = df[["ring-type","stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color",  "stem-height"]]
#y:
y = df["class"]
#%%

#%%
on_data=pd.get_dummies(X,drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(on_data, y, test_size=0.3 ,random_state=123)


#%%

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
#%%


tree.plot_tree(dtree)
plt.show()



#%%



y_pred=dtree.predict(X_test)

y_pred





# %%
