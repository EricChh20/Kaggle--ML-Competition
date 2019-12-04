import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from skimage.transform import resize

keras = tf.keras 


base_dir = './tiny_image_data/ukycs460g2019p1/'
IMG_SIZE = 160 # All images will be resized to 160x160
im_train = [] 
im_test = [] 

# load features, ignore header, ignore IDs
X_train = np.loadtxt(base_dir + 'train_features.csv', delimiter=',')[:,1:]
X_test = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:]
y_train = np.loadtxt(base_dir + 'train_labels.csv', dtype=np.uint8, delimiter=',', skiprows=1)[:,-1]
#print(X_train.shape)

# resizing our image data
for img in X_train:
    trainImg = img.reshape((30,30,3), order='F')
    testImg = img.reshape((30,30,3), order='F')
    train_res = resize(trainImg, (IMG_SIZE,IMG_SIZE,3))
    test_res = resize(testImg, (IMG_SIZE,IMG_SIZE,3))
    im_train.append(train_res)
    im_test.append(test_res)

# convert to np arrays
train_X = np.array(im_train)
test_X = np.array(im_test)

# our training data should have format (n, height, width, channel) for our desired TF model
# print(train_X.shape)
# print(y_train.shape)
# print(test_X.shape)


# plt.figure(1)
# plt.imshow(im_train/255)
# plt.axis('off')


# have images show in mac terminal
#plt.show()
