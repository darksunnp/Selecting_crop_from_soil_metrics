# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")
print(crops.head())

# Write your code here
X = crops.drop('crop', axis=1).values
y = crops['crop'].values
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=42, shuffle = True)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print("For all features")
print(f"Accuracy: {accuracy}")

X = crops['N'].values
X = X.reshape(-1, 1)
y = crops['crop'].values
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=42, shuffle = True)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print("N")
print(f"Accuracy: {accuracy}")

X = crops['P'].values
X = X.reshape(-1, 1)
y = crops['crop'].values
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=42, shuffle = True)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print("P")
print(f"Accuracy: {accuracy}")

X = crops['K'].values
X = X.reshape(-1, 1)
y = crops['crop'].values
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=42, shuffle = True)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_k = metrics.accuracy_score(y_test,y_pred)
print("K")
print(f"Accuracy: {accuracy_k}")

X = crops['ph'].values
X = X.reshape(-1, 1)
y = crops['crop'].values
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.9, random_state=42, shuffle = True)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print("ph")
print(f"Accuracy: {accuracy}")

best_predictive_feature = {"K":accuracy_k}
print(best_predictive_feature)
