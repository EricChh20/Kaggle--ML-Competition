import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt

base_dir = './tiny_image_data/ukycs460g2019p1/'
IMG_SIZE = 75 # All images will be resized to 160x160
im_test = [] 

# load features, ignore header, ignore IDs
X_test = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:]
#print(X_train.shape)

# resizing our image data
for img in X_train:
    testImg = img.reshape((30,30,3), order='F')
    test_res = resize(testImg, (IMG_SIZE,IMG_SIZE,3))
    im_test.append(test_res)
# convert to np arrays
test_X = np.array(im_test)

model.load_weights("model2.h5")

pred = model.predict(test_X)
max_value = max(pred[6])
max_index = np.where(pred[6]==max_value)

print(max_value)
print(max_index)