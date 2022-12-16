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
xf = df[['season' , 'habitat', 'does-bruise-or-bleed' ,'cap-color' ,'cap-diameter','ring-type', 'stem-width', 'cap-shape', 'gill-color', 'stem-height', 'stem-color', 'gill-attachment']]

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
