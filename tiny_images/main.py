import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf 

from keras.layers import Dense,GlobalAveragePooling2D,Flatten
from keras.applications import InceptionV3, MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from keras.optimizers import Adam
from skimage.transform import resize

from google.colab import files


#----------------------- PARAMETERS ----------------------------
IMG_SIZE = 160 # All images will be resized to 160x160
epochs = 100
im_test = [] 
im_train = [] 
sol = []

#----------------------- DATA HANDLING ----------------------------
train_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/train_features.csv'
train_label = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/train_labels.csv'
test_features = 'https://raw.githubusercontent.com/EricChh20/Kaggle-ML-Competition/master/tiny_images/tiny_image_data/ukycs460g2019p1/test_features.csv'
# load features, ignore header, ignore IDs
X_train = np.loadtxt(train_features, delimiter=',')[:,1:]
X_test = np.loadtxt(test_features, delimiter=',')[:,1:]
y_train = np.loadtxt(train_label, dtype=np.uint8, delimiter=',', skiprows=1)[:,-1]
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

x_train = train_X.astype('float32') / 255
x_test = train_X.astype('float32') / 255

# print(len(train_X))
# print(train_X[1].shape)
# print(len(test_X))
# print(test_X[1].shape)


#----------------------- MODEL CONFIG ----------------------------
base_model=MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
# base_model = InceptionV3(weights='imagenet',include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
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
# And 
sgd = optimizers.Adam(lr=0.001)
model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#----------------------- DATA AUGMENTATION ----------------------------
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split = .2)

#----------------------- TRAINING ----------------------------
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
# model.fit(train_X, y_train, epochs=epochs)
model.save_weights("mobilenet_tiny_img.h5")
#model.load_weights("mobilenet_tiny_img.h5")

# #----------------------- PREDICTIONS ----------------------------
pred = model.predict(test_X)
# print(pred[5])
for i in range(0,len(pred)):
    max_value = np.argmax(pred[i])
    sol.append(max_value)
    
print(sol)

#----------------------- CVS OUTPUT ----------------------------
df = pd.DataFrame(sol, columns=['Label'])
df.index += 1 # "upgrade" to one-based indexing
df.to_csv('knn_submission2.csv',index_label='ID',columns=['Label'])
