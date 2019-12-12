import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#----------------------- DATA HANDLING ----------------------------
train_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/train_features.csv'
train_label = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/train_labels.csv'
test_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/test_features.csv'
# load features, ignore header, ignore IDs
X_train = np.loadtxt(train_features, delimiter=',')[:,1:]
X_test = np.loadtxt(test_features, delimiter=',')[:,1:]
y_train = np.loadtxt(train_label, dtype=np.uint8, delimiter=',', skiprows=1)[:,-1]
#print(X_train.shape)

# X_train, X_test2, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#best found parameters from grid search
clf = RandomForestClassifier()
# perform grid search on best parameters 
# param_grid = {
#                  'n_estimators': [500, 1000, 1500, 2000, 2500, 3000],
#                  'max_depth': [1, 2, 5, 7, 9, 15],
#                  'max_features': [5, 15, 25, 50, 100, 150, 200]
#              }
# grid_clf = GridSearchCV(clf, param_grid)
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)

clf.fit(X_train, y_train)
df = pd.DataFrame(clf.predict(X_test), columns=['Label'])
df.index += 1 # "upgrade" to one-based indexing
df.to_csv('rf_submission.csv',index_label='ID',columns=['Label'])