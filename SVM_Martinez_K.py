'''SVM Model
By: Karina Martinez'''

#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

#%%
df = pd.read_csv('final.csv')
df.info()

#%%

X = df[["ring-type", "stem-width", "cap-shape", "gill-attachment", "stem-color", "gill-color", "stem-height"]]
#X = df.iloc[:,2:]  #all variables

y = df["class"]

numlist = ["stem-width", "stem-height"]
#numlist = ["stem-width", "stem-height","cap-diameter"] #all variables

catlist = X.columns.drop(numlist).to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=123)

#%%
X_Preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numlist),
        ("categorical1", OneHotEncoder(), catlist)
    ],
    verbose_feature_names_out=False,
    sparse_threshold=0
).fit(X_train)

#X_Preprocessor.get_feature_names_out()

#%%
X_train_std = pd.DataFrame(X_Preprocessor.fit_transform(X_train), columns=X_Preprocessor.fit(X_train).get_feature_names_out())
X_train_std.head()

#%%
y_Preprocessor = LabelBinarizer().fit(y_train)

y_train_std = pd.DataFrame(y_Preprocessor.fit_transform(y_train), columns = ["class"])
y_train_std.head()

#%%
model1 = svm.SVC()
model1.fit(X_train_std, np.ravel(y_train_std))

#%%
X_test_std = pd.DataFrame(X_Preprocessor.transform(X_test), columns=X_Preprocessor.fit(X_train).get_feature_names_out())
X_test_std.head()

y_test_std = pd.DataFrame(y_Preprocessor.transform(y_test), columns = ["class"])
y_test_std.head()

#%%
print(f'svc train score:  {model1.score(X_train_std,np.ravel(y_train_std))}')
print(f'svc test score:  {model1.score(X_test_std,np.ravel(y_test_std))}')

#%%
print(confusion_matrix(y_test_std, model1.predict(X_test_std)))
print(classification_report(y_test_std, model1.predict(X_test_std)))

#%%




