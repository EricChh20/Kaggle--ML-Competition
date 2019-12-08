import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf 

from keras.layers import Dense,GlobalAveragePooling2D,Flatten
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from skimage.transform import resize


base_dir = './tiny_image_data/ukycs460g2019p1/'
IMG_SIZE = 160 # All images will be resized to 160x160
im_test = [] 
im_train = [] 
sol = []

# load features, ignore header, ignore IDs
X_train = np.loadtxt(base_dir + 'train_features.csv', delimiter=',')[:,1:]
X_test = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:]
y_train = np.loadtxt(base_dir + 'train_labels.csv', dtype=np.uint8, delimiter=',', skiprows=1)[:,-1]
#print(X_train.shape)

# resizing our image data
for img in X_train:
    trainImg = img.reshape((30,30,3))
    train_res = resize(trainImg, (IMG_SIZE,IMG_SIZE,3))
    im_train.append(train_res)
    
for img in X_test:
    testImg = img.reshape((30,30,3))
    test_res = resize(testImg, (IMG_SIZE,IMG_SIZE,3))
    im_test.append(test_res)    

# convert to np arrays
train_X = np.array(im_train)
test_X = np.array(im_test)

print(len(train_X))
print(train_X[1].shape)
print(len(test_X))
print(test_X[1].shape)

# base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model = InceptionV3(weights='imagenet',include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(28,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_X, y_train, epochs=5)

model.save_weights("mobilenet_tiny_img.h5")
#model.load_weights("mobilenet_tiny_img.h5")

pred = model.predict(test_X)
# print(pred[5])

for i in range(0,len(pred)):
    max_value = np.argmax(pred[i])
    sol.append(max_value)
    
print(sol)
df = pd.DataFrame(sol, columns=['Label'])
df.index += 1 # "upgrade" to one-based indexing
df.to_csv('knn_submission2.csv',index_label='ID',columns=['Label'])
