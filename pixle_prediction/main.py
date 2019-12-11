import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor


full_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/pixle_prediction/roxie/roxie_full_features.csv'
test_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/pixle_prediction/roxie/roxie_test_features.csv'
train_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/pixle_prediction/roxie/roxie_train_features.csv'
train_values = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/pixle_prediction/roxie/roxie_train_values.csv'

X_train = np.loadtxt(train_features, delimiter=",")[:,1:]
X_test = np.loadtxt( test_features, delimiter=",")
ids_test = X_test[:,(0,)]
X_test = X_test[:,1:]
y_train = np.loadtxt(train_values, delimiter=",", ndmin=2)[:,(1,)]

# train model
mdl = DecisionTreeRegressor()
mdl.fit(X_train, y_train)

# make predictions on test data
y_pred = mdl.predict(X_test)

output = np.concatenate((ids_test, y_pred[:,np.newaxis]), axis=1)
np.savetxt("submission_decision_tree.csv", output, delimiter=",", fmt='%1.4f', header='ID,intensity')

# this file contains all pixels (the union of the train and test sets)
X_full = np.loadtxt(full_features, delimiter=",")

# make predictions for all pixels
y_pred = mdl.predict(X_full)

# show it as an image
plt.figure(figsize=(10,15))
plt.imshow(y_pred.reshape((650,430,3)))