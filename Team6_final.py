#!/usr/bin/env python

"""Team6_final.py: Mushroom Edibility Classification Final Project"""

__author__ = "Caroline Krall, Jerrick Gerald, Karina Martinez, Lijia Ren"

#%% [markdown]
# # Mushroom Edibility Classification
# By: Caroline Krall, Jerrick Gerald, Karina Martinez, Lijia Ren (Team 6)
#
# ## Data Cleaning

#%%
import pandas as pd
import numpy as np

#%%
# Importing data
data = pd.read_csv('secondary_data.csv',sep=';')
data.head()

#%%
data['class'].value_counts()

#%%
data.info()

#%%
missing_values_count = data.isnull().sum()
missing_values_count

#%%
# Drop columns with high missing count
data=data.drop(['cap-surface','gill-spacing','stem-root','stem-surface','veil-type','veil-color','spore-print-color'],axis=1)
data.head()

#%%
# Fill remaining missing values
data= data.fillna(0)
data.isnull().sum()

#%%
data.head()

# %%
# Export to new file
# data.to_csv('final.csv')

# %% [markdown]
# ## EDA
# ### Distribution Plots

#%%
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv('final.csv')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

# %%
# Cap Shape Distribution
sns.displot(df, x="cap-shape", hue="class", multiple="stack", palette="husl").set(title="Cap Shape Distribution")
plt.show()

#%%
table1 = pd.crosstab(df['cap-shape'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%%
# Cap Color Distribution
sns.displot(df, x="cap-color", hue="class", multiple="stack", palette="husl").set(title="Cap Color Distribution")
plt.show()

#%%
table1 = pd.crosstab(df['cap-color'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%%
# Gill Attachment Distribution
sns.displot(df, x="gill-attachment", hue="class", multiple="stack", palette="husl").set(title="Gill Attachment Distribution")
plt.show()

#%%
table1 = pd.crosstab(df['gill-attachment'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

#%%
# Gill Color Distribution
sns.displot(df, x="gill-color", hue="class", multiple="stack", palette="husl").set(title="Gill Color Distribution")
plt.show()

#%%
table1 = pd.crosstab(df['gill-color'], 
                            df['class'],
                                margins = False)

table1['total'] = table1.sum(axis=1)
table1['% e'] = (table1['e'] / table1.total * 100).round(2)
table1['% p'] = (table1['p'] / table1.total * 100).round(2)
table1.sort_values('total', ascending=False)

# %%
# Does-Bruise-Or-Bleed Distribution
sns.displot(df, x="does-bruise-or-bleed", hue="class", multiple="stack", palette="husl").set(title="Does-Bruise-Bleed Distribution")
plt.show()

#%%
table2 = pd.crosstab(df['does-bruise-or-bleed'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

#%%
# Season Distribution
sns.displot(df, x="season", hue="class", multiple="stack", palette="husl").set(title="Season Distribution")
plt.show()

#%%
table2 = pd.crosstab(df['season'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

# %%
# Habitat Distribution
sns.displot(df, x="habitat", hue="class", multiple="stack", palette="husl").set(title="Habitat Distribution")
plt.show()

#%%
table2 = pd.crosstab(df['habitat'], 
                            df['class'],
                                margins = False)

table2['total'] = table2.sum(axis=1)
table2['% e'] = (table2['e'] / table2.total * 100).round(2)
table2['% p'] = (table2['p'] / table2.total * 100).round(2)
table2.sort_values('total', ascending=False)

#%%
# Stem Width Distribution
sns.histplot(df, x="stem-width", hue="class", multiple="stack", palette="husl", bins=20).set(title="Stem Width Distribution")
plt.show()

#%%
# Stem Width Box-Plot
sns.boxplot(y='stem-width',x='class',data=df, palette='husl').set(title="Stem Width Distribution")
plt.show()

#%%
# Stem Height
# plt.hist(df['stem-height'], stacked = True)
# df.pivot(columns='class', values = 'stem-height').plot.hist(stacked = True)

# Stem Height Distribution
sns.histplot(df, x="stem-height", hue="class", multiple="stack", palette="husl", bins=20).set(title="Stem Height Distribution")
plt.show()

# %%
# Stem Height Box-Plot
sns.boxplot(y='stem-height',x='class',data=df, palette='husl').set(title="Stem Height Distribution")
plt.show()

#%%
# Has-ring Distribution
# counts = df['has-ring'].value_counts()
# df.groupby(['has-ring']).size().plot(kind = "bar")
# df.groupby(['has-ring', 'class']).size().plot(kind = "bar", stacked=True)
# print(counts)

#%%
# Has Ring Distribution
sns.displot(df, x="has-ring", hue="class", multiple="stack", palette="husl").set(title="Has-Ring Distribution")
plt.show()

#%%
table3 = pd.crosstab(df['has-ring'], 
                            df['class'],
                                margins = False)

table3['total'] = table3.sum(axis=1)
table3['% e'] = (table3['e'] / table3.total * 100).round(2)
table3['% p'] = (table3['p'] / table3.total * 100).round(2)
table3.sort_values('total', ascending=False)

# %%
# Ring type
# df.groupby(['ring-type']).size().plot(kind = "bar", stacked=True)
# counts = df['ring-type'].value_counts()
# print(counts)

# pd.crosstab(df['ring-type'], df['class']).plot(kind='bar', stacked=True)

#%%
# Ring Type Distribution
sns.displot(df, x="ring-type", hue="class", multiple="stack", palette="husl").set(title="Ring Type Distribution")
plt.show()

#%%
table3 = pd.crosstab(df['ring-type'], 
                            df['class'],
                                margins = False)

table3['total'] = table3.sum(axis=1)
table3['% e'] = (table3['e'] / table3.total * 100).round(2)
table3['% p'] = (table3['p'] / table3.total * 100).round(2)
table3.sort_values('total', ascending=False)

#%%
# cap-diameter
# plt.hist(df['cap-diameter'])
# df.pivot(columns='class', values = 'cap-diameter').plot.hist(stacked = True)

#%%
# Cap Diameter Distribution
sns.histplot(df, x="cap-diameter", hue="class", multiple="stack", palette="husl", bins=20).set(title="Cap Diameter Distribution")
plt.show()

# %%
# Cap Diameter Box-Plot
sns.boxplot(y='cap-diameter',x='class',data=df, palette='husl').set(title="Cap Diameter Distribution")
plt.show()

#%%
# Stem Color Distribution
# count_classes = pd.value_counts(df['stem-color'], sort = True)
# count_classes.plot(kind = 'bar', rot=0,color='tan')

#%%
# Stem Color Distribution
sns.displot(df, x="stem-color", hue="class", multiple="stack", palette="husl").set(title="Stem Color Distribution")
plt.show()

#%%
table5 = pd.crosstab(df['stem-color'], 
                            df['class'],
                                margins = False)

table5['total'] = table5.sum(axis=1)
table5['% e'] = (table5['e'] / table5.total * 100).round(2)
table5['% p'] = (table5['p'] / table5.total * 100).round(2)
table5.sort_values('total', ascending=False)

#%%
# Class Distribution
# count_classes = pd.value_counts(df['class'], sort = True)
# count_classes.plot(kind = 'bar', rot=0)

#%%
# Class Distribution
sns.displot(df, x="class", hue= "class", palette="husl").set(title="Class Distribution")
plt.show()

#%%
table4 = pd.DataFrame(df['class'].value_counts())
table4

#%% [markdown]
# ### Correlation Plots

#%%
# Correlation Matrix
df1 = df.copy(deep=True)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df1[column] = labelencoder.fit_transform(df1[column])

comat=df1.corr()
k=15
col=comat.nlargest(k,'class')['class'].index
cm=np.corrcoef(df1[col].values.T)

plt.figure(figsize=(16, 10))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,annot=True,cbar=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=col.values, xticklabels=col.values)
plt.show()

#%%
# Stem Width vs Cap Diameter
sns.scatterplot(data=df, x="cap-diameter", y="stem-width", hue="class", palette="husl").set(title="Stem Width vs Cap Diameter")
plt.show()

#%%
# Stem Height vs Cap Diameter
sns.scatterplot(data=df, x="cap-diameter", y="stem-height", hue="class", palette="husl").set(title="Stem Height vs Cap Diameter")
plt.show()

#%%
# Stem Width vs Stem Height
sns.scatterplot(data=df, x="stem-height", y="stem-width", hue="class", palette="husl").set(title="Stem Width vs Stem Height")
plt.show()

#%%
#%%
# Ring Type Distribution by Has-Ring
sns.displot(df, x="ring-type", hue= "has-ring", multiple="stack", palette="husl").set(title="Ring Type Distribution by Has-Ring")
plt.show()

#%% [markdown]
# ## Models
# ### Logistic Regression

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("final.csv")
print(df.head())
#%%
#need to encode all strings to floats
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass

print(df.head())
# NOTE: class p = 1

# %%
# based on the model summary, I will drop gill attachment and stem color from the model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
yf = df[['class']]
xf = df[['ring-type', 'stem-width', 'cap-shape', 'gill-color', 'stem-height', 'stem-color', 'gill-attachment']]

xftrain, xftest, yftrain, yftest = train_test_split(xf, yf, test_size=0.30, random_state=123)
fullLogit = LogisticRegression() #initiate logit model

fullLogit.fit(xftrain, yftrain)

#%%
from sklearn.metrics import classification_report
print(fullLogit.score(xftest, yftest))
print(fullLogit.score(xftrain, yftrain))

yftrue, yfpred = yftest, fullLogit.predict(xftest)
print(classification_report(yftrue, yfpred))

# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf = confusion_matrix(yftrue, yfpred)
ConfusionMatrixDisplay.from_predictions(yftrue, yfpred)

# %%
# Model Evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

print(precision_score(yftest, yfpred))
print(recall_score(yftest, yfpred))
print(accuracy_score(yftest, yfpred))
print('F1 Score: %.3f' % f1_score(yftest, yfpred))
print(roc_auc_score(yftest, yfpred))
# %%
# ROC Curve chart
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(yftest, yfpred)

#%%
# reduced model
y = df[['class']]
x = df[['ring-type', 'stem-width', 'cap-shape', 'gill-color', 'stem-height']]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=123)
fullLogit = LogisticRegression() #initiate logit model

fullLogit.fit(xtrain, ytrain)


#%%
# model score and classification report
from sklearn.metrics import classification_report
print(fullLogit.score(xtest, ytest))
print(fullLogit.score(xtrain, ytrain))

ytrue, ypred = ytest, fullLogit.predict(xtest)
print(classification_report(ytrue, ypred))

# %%
# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
conf = confusion_matrix(ytrue, ypred)
ConfusionMatrixDisplay.from_predictions(ytrue, ypred)

# %%
# Model Evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

print(precision_score(ytest, ypred))
print(recall_score(ytest, ypred))
print(accuracy_score(ytest, ypred))
print('F1 Score: %.3f' % f1_score(ytest, ypred))
print(roc_auc_score(ytest, ypred))
# %%
# ROC Curve chart
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(ytest, ypred)


# %%
#%% [markdown]
# ### Decision Tree

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


#%%
df.info()
#%%
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

#%% [markdown]
# ### Random Forest

#%%
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


#%%
df=pd.read_csv(r'final.csv')
df=df.drop(['Unnamed: 0'],axis=1)

#%%
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

#%%
features=["ring-type", "stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height","stem-width", "stem-height"]
prediction=['class']

#%%
X=df[features]
y=df[prediction]

#%%
#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)

#%%
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#%%
print("Random Forest Classifier ACCURACY:",metrics.accuracy_score(y_test,y_pred))

#%%
print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))

#%%
model = RandomForestClassifier(max_depth=2, n_estimators=30,min_samples_split=3, max_leaf_nodes=5,random_state=22)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Training Accuracy : ',metrics.accuracy_score(y_train,model.predict(X_train))*100)
print('Validation Accuracy : ',metrics.accuracy_score(y_test,model.predict(X_test))*100)

#%%
print("Random Forest Classifier:\n",metrics.classification_report(y_test,y_pred))

#%%
cf_mat = confusion_matrix(y_test, y_pred)
print(cf_mat)
sns.heatmap(cf_mat/np.sum(cf_mat), annot=True, 
            fmt='.2%', cmap='Blues')

#%%
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

#%%
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

plt.legend()

#%%
# AUC - ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
plt.plot(fpr,tpr,label="Gradient Boosting, AUC="+str(auc))

#add legend
plt.legend()

#%% [markdown]
# ### SVM

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import RocCurveDisplay

#%%
# Read in the Mushroom Edibility dataset
df = pd.read_csv('final.csv')
df.info()

#%%
# Define predictors and target variables
# X = df.drop(['Unnamed: 0', 'class'], axis=1) #all predictors
X = df.drop(['Unnamed: 0', 'class', 'has-ring'], axis=1) # exclude has-ring
y = df["class"]

# Numerical predictors
numlist = ["stem-width", "stem-height","cap-diameter"] #all predictors

# Categorical predictors
catlist = X.columns.drop(numlist).to_list()

#%%
# Split and Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=123)

#%%
# Standardization and binarization of target
X_Preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numlist),
        ("categorical1", OneHotEncoder(), catlist)
    ],
    verbose_feature_names_out=False,
    sparse_threshold=0
)

X_train_std = pd.DataFrame(X_Preprocessor.fit_transform(X_train), columns=X_Preprocessor.fit(X_train).get_feature_names_out())
#X_train_std.head()
X_test_std = pd.DataFrame(X_Preprocessor.transform(X_test), columns=X_Preprocessor.fit(X_test).get_feature_names_out())
#X_test_std.head()

y_Preprocessor = LabelBinarizer().fit(y_train)
y_train_std = pd.DataFrame(y_Preprocessor.fit_transform(y_train), columns = ["class"])
#y_train_std.head()
y_test_std = pd.DataFrame(y_Preprocessor.transform(y_test), columns = ["class"])
#y_test_std.head()

#%%
# SVM Model w/ linear kernel
model1 = svm.SVC(kernel='linear')
model1.fit(X_train_std, np.ravel(y_train_std))

print(f'svc train score:  {model1.score(X_train_std,np.ravel(y_train_std))}') # 0.7803639936371293

#%%
# SVM Model w/ RBF kernel - Default parameters
model2 = svm.SVC() # C=1, gamma='scale' = 0.08510638297872342
model2.fit(X_train_std, np.ravel(y_train_std))

print(f'svc train score:  {model2.score(X_train_std,np.ravel(y_train_std))}') # 0.99319266398428

#%%
# Hyperparameter grid search - Takes ~90 mins to run!!
# Defining parameter range
param_grid = {'C': [0.1, 1, 10], 
              'gamma': [0.01, 1, 10],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3, cv=3)
  
# fitting the model for grid search
grid.fit(X_train_std, np.ravel(y_train_std))

# print best parameter after tuning
print(grid.best_params_) #{'C': 10, 'gamma': 1, 'kernel': 'rbf'}
  
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_) #SVC(C=10, gamma=1)

#%%
# Best SVM Model
bestsvm = svm.SVC(C=10, gamma=1)
bestsvm.fit(X_train_std, np.ravel(y_train_std))

#%%
# Train score
print(f'svc train score:  {bestsvm.score(X_train_std,np.ravel(y_train_std))}') #0.9988303546364742

# Mean cross validated score
# scores = cross_val_score(bestsvm, X_train_std, np.ravel(y_train_std), cv=5)
# print(f'svc cv scores: {scores}')
# print(f'mean svc cv score: {scores.mean()}')
#svc cv scores: [0.99836257 0.99812865 0.99871345 0.99719265 0.99801146]
#mean svc cv score: 0.9980817591606419

#%%
# Model test set score
print(f'svc test score:  {bestsvm.score(X_test_std,np.ravel(y_test_std))}') # 0.998526281316522

#%%
# Confusion Matrix
#print(confusion_matrix(y_test_std, bestsvm.predict(X_test_std)))
#[[ 8132    23]
 #[    4 10162]]

y_pred = bestsvm.predict(X_test_std)
conf_matrix = confusion_matrix(y_true=y_test_std, y_pred=y_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#%%
# Classification Report
print('Precision: %.3f' % precision_score(y_test_std, y_pred)) # Precision: 0.998
print('Recall: %.3f' % recall_score(y_test_std, y_pred)) # Recall: 1.000
print('Accuracy: %.3f' % accuracy_score(y_test_std, y_pred)) # Accuracy: 0.999
print('F1 Score: %.3f' % f1_score(y_test_std, y_pred)) # F1 Score: 0.999

#print(classification_report(y_test_std, bestsvm.predict(X_test_std)))
"""
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      8155
           1       1.00      1.00      1.00     10166

    accuracy                           1.00     18321
   macro avg       1.00      1.00      1.00     18321
weighted avg       1.00      1.00      1.00     18321
"""

#%%
# ROC/AUC Plot
svc_disp = RocCurveDisplay.from_estimator(bestsvm, X_test_std, y_test_std)
plt.show()
# SVC(AUC=1.00)

#%%





