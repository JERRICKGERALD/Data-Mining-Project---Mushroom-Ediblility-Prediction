#!/usr/bin/env python

"""SVM_Martinez_K.py: SVM model for Mushroom Edibility Classification dataset"""

__author__ = "Karina Martinez"

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
X = df.drop(['Unnamed: 0', 'class'], axis=1) #all predictors
#X = df[["ring-type", "stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height"]]
y = df["class"]

# Numerical predictors
numlist = ["stem-width", "stem-height","cap-diameter"] #all predictors
#numlist = ["stem-width", "stem-height"]

# Categorical predictors
catlist = X.columns.drop(numlist).to_list()

#%%
# Split and Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=123)

X_Preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numlist),
        ("categorical1", OneHotEncoder(), catlist)
    ],
    verbose_feature_names_out=False,
    sparse_threshold=0
).fit(X_train)

#%%
# Standardization and binarization of training set
X_train_std = pd.DataFrame(X_Preprocessor.fit_transform(X_train), columns=X_Preprocessor.fit(X_train).get_feature_names_out())
#X_train_std.head()

y_Preprocessor = LabelBinarizer().fit(y_train)
y_train_std = pd.DataFrame(y_Preprocessor.fit_transform(y_train), columns = ["class"])
#y_train_std.head()

#%%
# SVM Model w/ linear kernel
model1 = svm.SVC(kernel='linear')
model1.fit(X_train_std, np.ravel(y_train_std))

print(f'svc train score:  {model1.score(X_train_std,np.ravel(y_train_std))}') # 0.7862356133620286

#%%
# SVM Model w/ RBF kernel - Default parameters
model2 = svm.SVC() # C=1, gamma='scale' = 0.08510638297872342
model2.fit(X_train_std, np.ravel(y_train_std))

print(f'svc train score:  {model2.score(X_train_std,np.ravel(y_train_std))}') # 0.9928183774679518

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
scores = cross_val_score(bestsvm, X_train_std, np.ravel(y_train_std), cv=5)
print(f'svc cv scores: {scores}')
print(f'mean svc cv score: {scores.mean()}')
#svc cv scores: [0.99836257 0.99812865 0.99871345 0.99719265 0.99801146]
#mean svc cv score: 0.9980817591606419

#%%
# Standardization of test set
X_test_std = pd.DataFrame(X_Preprocessor.transform(X_test), columns=X_Preprocessor.fit(X_train).get_feature_names_out())
X_test_std.head()

y_test_std = pd.DataFrame(y_Preprocessor.transform(y_test), columns = ["class"])
y_test_std.head()

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


