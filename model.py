import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder





df = pd.read_csv('data/music.csv')
X = df.drop(columns=['genre'])
y = df['genre']

categorical_column = 'gender'
label_encoder = LabelEncoder()
X[categorical_column] = label_encoder.fit_transform(X[categorical_column])

# X_categorical = pd.get_dummies(X[categorical_column])

# X_encoded = pd.concat([X_categorical, X.drop(columns=[categorical_column])], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

y_train_encoded = pd.get_dummies(y_train)
y_train = np.array(y_train).reshape(-1, 1)

encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train).toarray()

y_test_encoded = pd.get_dummies(y_test)
y_test = np.array(y_test).reshape(-1, 1)
y_test_encoded = encoder.transform(y_test).toarray()
model= DecisionTreeClassifier()
model.fit(X_train, y_train_encoded)
joblib.dump(model, 'musicmodel.joblib')
joblib.dump(encoder, 'Encoder.joblib')
joblib.dump(label_encoder,'labelencoder.joblib')