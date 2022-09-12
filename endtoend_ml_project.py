import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

import pickle

df = pd.read_csv("placement.csv")
# print(df.head())

df = df.iloc[:,1:]

# print(df.head())

plt.scatter(df['cgpa'], df['iq'] ,c=df['placement'])
# plt.show()

X=  df.iloc[:, 0:2]
Y = df.iloc[:,-1]
# print(X)
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# print(X_train)
# print(Y_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
# print(X_train)

X_test = scaler.transform(X_test)
# print(X_test)

clf = LogisticRegression()

# Model Training
clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)

#  Accuracy is 90 percent

plot_decision_regions(X_train, Y_train.values, clf=clf, legend=2)
plt.show()

pickle.dump(clf, open('model.pkl', 'wb'))