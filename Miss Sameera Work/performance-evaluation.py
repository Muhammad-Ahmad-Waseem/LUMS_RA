# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:40:59 2022

@author: Sohaib Bin Mohsin
"""

import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy
from scipy.signal import welch, detrend, filtfilt
import pickle

lowcut = 0.5
highcut = 40.0
nyq = 0.5 * 256
low = lowcut/nyq
high = highcut/nyq
b, a = scipy.signal.butter(2, [low, high], 'bandpass', analog=False)

left_points = pd.read_csv("Data/Left.csv").to_numpy()[:, 1].reshape(256, -1).T
left_points = scipy.signal.filtfilt(b, a, left_points, axis=0)
left_points = detrend(left_points)
# left_points = np.transpose(left_points, (1, 0, 2)).reshape(-1, 256*4)
left_labels = np.full((left_points.shape[0]), 0)

right_points = pd.read_csv("Data/Right.csv").to_numpy()[:, 1].reshape(256, -1).T
right_points = scipy.signal.filtfilt(b, a, right_points, axis=0)
right_points = detrend(right_points)
# right_points = np.transpose(right_points, (1, 0, 2)).reshape(-1, 256*4)
right_labels = np.full((right_points.shape[0]), 1)

blink_points = pd.read_csv("Data/Blink.csv").to_numpy()[:, 1].reshape(256, -1).T
blink_points = scipy.signal.filtfilt(b, a, blink_points, axis=0)
blink_points = detrend(blink_points)
# blink_points = np.transpose(blink_points, (1, 0, 2)).reshape(-1, 256*4)
blink_labels = np.full((blink_points.shape[0]), 2)

print(left_points.shape)
print(right_points.shape)
print(blink_points.shape)

X = np.vstack([left_points, right_points, blink_points])
# X = np.extend(X, right_points)
# X = np.extend(X, blink_points)

Y = left_labels
# Y = np.append([left_labels, right_labels, blink_labels])
Y = np.append(Y, right_labels)
Y = np.append(Y, blink_labels)

print(X.shape)
print(Y.shape)

# print(Y[60])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# print(X_train.shape)
# print(X_test.shape)
# print(y_test)


# Normalize the train data for numerical stability
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

# Normalize the test data for numerical stability
ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

# Initializing each binary classifier
models = {}

# Logistic Regression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
models['K-Nearest Neighbor'] = KNeighborsClassifier()

# Performance evaluation of each binary classifier
# Blink classifier
accuracy, precision, recall = {}, {}, {}

for key in models.keys():

    # Fit the classifier
    models[key].fit(X_train, y_train)

    # Make predictions
    predictions = models[key].predict(X_test)

    print(key)
    print(predictions)
    print(y_test)
    print()

    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test, average='micro')
    recall[key] = recall_score(predictions, y_test, average='micro')

    filename = '{}.sav'.format(key.replace(' ', '_'))
    pickle.dump(models[key], open(filename, 'wb'))

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model.to_csv('Blink-classifier-performance-evaluation2.csv', mode='w')
#
# # Left classifier
# accuracy2, precision2, recall2 = {}, {}, {}
#
# for key in models.keys():
#
#     # Fit the classifier
#     models[key].fit(X_train2, y_train2)
#
#     # Make predictions
#     predictions = models[key].predict(X_test2)
#
#     # Calculate metrics
#     accuracy2[key] = accuracy_score(predictions, y_test2)
#     precision2[key] = precision_score(predictions, y_test2)
#     recall2[key] = recall_score(predictions, y_test2)
#
# df_model2 = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
# df_model2['Accuracy'] = accuracy2.values()
# df_model2['Precision'] = precision2.values()
# df_model2['Recall'] = recall2.values()
# df_model2.to_csv('Left-classifier-performance-evaluation1.csv', mode='w')
#
# # Right classifier
# accuracy3, precision3, recall3 = {}, {}, {}
#
# for key in models.keys():
#
#     # Fit the classifier
#     models[key].fit(X_train3, y_train3)
#
#     # Make predictions
#     predictions = models[key].predict(X_test3)
#
#     # Calculate metrics
#     accuracy3[key] = accuracy_score(predictions, y_test3)
#     precision3[key] = precision_score(predictions, y_test3)
#     recall3[key] = recall_score(predictions, y_test3)
#
# df_model3 = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
# df_model3['Accuracy'] = accuracy3.values()
# df_model3['Precision'] = precision3.values()
# df_model3['Recall'] = recall3.values()
# df_model3.to_csv('Right-classifier-performance-evaluation1.csv', mode='w')
