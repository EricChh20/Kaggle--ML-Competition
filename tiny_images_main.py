import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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

# have images show in mac terminal
#plt.show()

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
# Create the base model from the pre-trained model MobileNet V2

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(28, activation='softmax'))

base_learning_rate = 0.0001
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.summary()
history = model.fit(train_X, y_train, epochs=20)

plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

train_loss, train_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(train__acc) 
