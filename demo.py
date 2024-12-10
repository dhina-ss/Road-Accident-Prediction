import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.DataFrame(pd.read_csv('train_final.csv'))

df.isnull().any().sum()

df = df.dropna()

X = df.drop('Accident_Severity', axis=1)
y = df['Accident_Severity']

# print(X)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dtree= DecisionTreeClassifier()

dtree.fit(X_train,y_train)

# predictions= dtree.predict([[]])

joblib.dump(dtree,'accident.pkl')

# print(predictions)

# print('Accuracy:', accuracy_score(y_test,predictions))
