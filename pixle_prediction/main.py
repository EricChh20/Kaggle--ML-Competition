import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV


X_train = np.loadtxt('roxie_train_features.csv', delimiter=",")[:,1:]
X_test = np.loadtxt( 'roxie_test_features.csv', delimiter=",")
ids_test = X_test[:,(0,)]
X_test = X_test[:,1:]
y_train = np.loadtxt('roxie_train_values.csv', delimiter=",", ndmin=2)[:,(1,)]

clf = MLPRegressor(hidden_layer_sizes=(100,50,50,20), solver='adam', activation='tanh')
# perform grid search on best parameters 
# param_grid = {
#                  'hidden_layer_sizes': [(100), (100,50,50,20)],
#                  'activation': ['tanh', 'relu'],
#                  'solver': ['sgd', 'adam'],
#                  'learning_rate': ['constant', 'adaptive']
#              }
# grid_clf = GridSearchCV(clf, param_grid)
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

output = np.concatenate((ids_test, y_pred[:,np.newaxis]), axis=1)
np.savetxt("submission_decision_tree.csv", output, delimiter=",", fmt='%1.4f', header='ID,intensity')
# this file contains all pixels (the union of the train and test sets)
# X_full = np.loadtxt('roxie_full_features.cvs', delimiter=",")
# # make predictions for all pixels
# y_pred = mdl.predict(X_full)

# # show it as an image
# plt.figure(figsize=(10,15))
# plt.imshow(y_pred.reshape((650,430,3)))