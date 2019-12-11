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
from keras.optimizers import Adam, SGD
from skimage.transform import resize
from keras.utils import to_categorical
from google.colab import files
from sklearn.model_selection import train_test_split
# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 
  

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
# onehotencoder = OneHotEncoder(categorical_features = [0]) 
# y_train = onehotencoder.fit_transform(data).toarray() 
# convert to np arrays
train_X = np.array(im_train)
test_X = np.array(im_test)
y_train = keras.utils.to_categorical(y_train, num_classes)
# X_train, X_test2, y_train, y_test = train_test_split(train_X, y_train, test_size=0.2, random_state=42)

#----------------------- DATA AUGMENTATION ----------------------------
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split = .2)
# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

#----------------------- Model ----------------------------
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(28, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#----------------------- TRAINING ----------------------------
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# train the model on the new data for a few epochs
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_X, y_train, batch_size=32),
                    steps_per_epoch=len(train_X) / 32, epochs=10)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['acc'])
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(train_X, y_train, batch_size=32),
                    steps_per_epoch=len(train_X) / 32, epochs=epochs)


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

files.download('knn_submission2.csv')
files.download('mobilenet_tiny_img.h5')