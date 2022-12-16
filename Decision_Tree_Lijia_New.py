# Decision Tree
# By: Lijia Ren



#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
#%%

# read data frame
df = pd.read_csv('final.csv')
df.info()
print(df.head())

#%%

#full model


#define Xf:

Xf = df[['season' , 'habitat', 'does-bruise-or-bleed' ,'cap-color' ,'cap-diameter','ring-type', 'stem-width', 'cap-shape', 'gill-color', 
'stem-height', 'stem-color', 'gill-attachment']]
#define yf:
yf = df["class"]


#%% 
# one_hot_data
one_hot_data=pd.get_dummies(Xf,drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(one_hot_data, yf, test_size=0.3 ,random_state=123)

#%%
# use a standard scale the features for processing
scaler=StandardScaler()
scale=scaler.fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)

#%%
# decision tree
dtree = DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=3)
dtree = dtree.fit(X_train, y_train)
#%%
#  plot of decision tree

tree.plot_tree(dtree)
plt.show()



#%%

# make prediction

y_pred=dtree.predict(X_test)

y_pred

#%%
# find accuracy score
score=accuracy_score(y_test, y_pred)
score


#%%
#creating a confusion matrix
confusion_matrix(y_test,y_pred)

#%%
#extracting TN, TP, FP, FN
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
(tn, fp, fn, tp)
# %%
# run classification report
conf_matrix= classification_report(y_test,y_pred)
print('Classification report; \n', conf_matrix)



print(confusion_matrix(y_test, dtree.predict(X_test)))
#%%

# make Confusion Matrix
y_pred = dtree.predict(X_test)
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

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
def plot_roc_curve(y_test, y_pred): 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
# %%
#find AUC score

from sklearn.metrics import RocCurveDisplay
dtree_disp = RocCurveDisplay.from_estimator(dtree, X_test, y_test)
plt.show()




#%%

# Reduced model
#define X:
features=["ring-type","stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color",  "stem-height"]
X = df[["ring-type","stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color",  "stem-height"]]
#define y:
y = df["class"]


#%% 
# one_hot_data
one_hot_data=pd.get_dummies(X,drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(one_hot_data, y, test_size=0.3 ,random_state=123)

#%%
# use a standard scale the features for processing
scaler=StandardScaler()
scale=scaler.fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)

#%%
# decision tree
dtree = DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=3)
dtree = dtree.fit(X_train, y_train)
#%%
#  plot of decision tree

tree.plot_tree(dtree)
plt.show()



#%%

# make prediction

y_pred=dtree.predict(X_test)

y_pred

#%%
# find accuracy score
score=accuracy_score(y_test, y_pred)
score


#%%
#creating a confusion matrix
confusion_matrix(y_test,y_pred)

#%%
#extracting TN, TP, FP, FN
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
(tn, fp, fn, tp)
# %%
# run classification report
conf_matrix= classification_report(y_test,y_pred)
print('Classification report; \n', conf_matrix)



print(confusion_matrix(y_test, dtree.predict(X_test)))
#%%

# make Confusion Matrix
y_pred = dtree.predict(X_test)
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

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
def plot_roc_curve(y_test, y_pred): 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
# %%
#find AUC score

from sklearn.metrics import RocCurveDisplay
dtree_disp = RocCurveDisplay.from_estimator(dtree, X_test, y_test)
plt.show()


