from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

Data_set = pd.read_csv("bodyPerformance.csv", delimiter=";")

Features_names = Data_set.columns[0:11]

print(Features_names)

target = Data_set['class'].tolist()


target = list(set(target))

print(target)

X = Data_set[Features_names].values


print(X)

y = Data_set["class"]


print(y)

# Data Preprocessing


label_gender = preprocessing.LabelEncoder()

label_gender.fit(['F', 'M'])
X[:, 1] = label_gender.transform(X[:, 1])


print(X)


print(Data_set.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=7)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


################ KNN###################


neigh = KNeighborsClassifier(n_neighbors=115)

neigh.fit(X_train, y_train)


predicted = neigh.predict(X_test)

print("\nPredicted by KNN", predicted)

results = confusion_matrix(y_test, predicted)

print("\n KNN confusion matrix", results)

print("\nKNN Accuracy: ", metrics.accuracy_score(y_test, predicted))


################# Naive#################

# create a GaussianNB Classifier

model = GaussianNB()

# train Model using Training Sets

model.fit(X_train, y_train)


predicted = model.predict(X_test)

print("\nPredicted by Naive", predicted)

results = confusion_matrix(y_test, predicted)
print("\n Naive confusion matrix", results)

print("\nNaive  Accuracy: ", metrics.accuracy_score(y_test, predicted))
