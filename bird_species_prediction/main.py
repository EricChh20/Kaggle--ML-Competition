import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP

X_train = np.loadtxt('train_features.csv', skiprows=1, delimiter=',')
X_test = np.loadtxt('test_features.csv', skiprows=1, delimiter=',')

y_train = np.loadtxt('train_values.csv', skiprows=1, delimiter=',').astype('int32')

# handle ID column
X_train = X_train[:,1:]
y_train = y_train[:,1:]
X_test_labels = X_test[:,(0,)]
X_test = X_test[:,1:]


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(100, 100, 50, 20), activation='tanh', solver='adam')
# perform grid search on best parameters 
# param_grid = {
#                  'hidden_layer_sizes': [(100,100,50,20), (50,50,20), (200,100,100,50)],
#                  'activation': ['tanh', 'relu'],
#                  'solver': ['sgd', 'adam']
#              }

# grid_clf = GridSearchCV(clf, param_grid)
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)

feats = (0,1,3) # only use location features
clf.fit(X_train[:,feats],y_train.flatten())
print(clf.score(X_train[:,feats],y_train))

#y_preds = clf.predict_proba(X_test)
y_preds = clf.predict(X_test[:,feats])[:,np.newaxis]

output = np.concatenate((X_test_labels,y_preds), axis=1)
np.savetxt('mlp_bird2.csv', output, header='ID,speciesKey', fmt='%i', delimiter=',')