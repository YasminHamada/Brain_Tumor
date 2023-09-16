import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
import keras.layers as layers
from keras.models import Model
import cv2
import pickle


data_path = 'brain_tumor_dataset'

# Data Preprocessing

data = tf.keras.utils.image_dataset_from_directory(data_path) # Generate Data
data = data.map(lambda x,y: (x/255, y))  # data normalization

# split data to train , validation and test
train_size = int(len(data)*.6)
val_size = int(len(data)*.3)
test_size = int(len(data)*.3)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Visualization Sample of Data

def brain_image(dataset):
  plt.figure(figsize=(12, 8))
  index = 0
  for image, label in dataset.take(15):
    index +=1
    ax = plt.subplot(4, 4, index)
    plt.imshow(image[index].numpy().astype("uint8"))
    plt.title(int(label[index]))
    plt.axis("off")

dataset =  tf.keras.utils.image_dataset_from_directory(data_path)

brain_image(dataset)

# Model Structure

# Load the pre-trained VGG16 model
vgg_model = VGG16(
weights='imagenet',
include_top=False,
input_shape=(256,256,3)
)

# Create a new model by adding a few layers on top of the pre-trained model
model = Sequential()
model.add(vgg_model)

model.add(Conv2D(16, (3,3),1, activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3),1, activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3),1, activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Freeze the weights of the pre-trained model
model.layers[0].trainable = False

# Compile the model with appropriate loss function, optimizer and metrics
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_vgg = model.fit(
    train,
    epochs=15,
    validation_data=val,
    callbacks=[early_stopping]
)

# Digram Visualization

fig = plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(history_vgg.history['loss'], color='teal', label='loss')
plt.plot(history_vgg.history['val_loss'], color='orange', label='val_loss')
plt.title('loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history_vgg.history['accuracy'], color='teal', label='Accuracy')
plt.plot(history_vgg.history['val_accuracy'], color='orange', label='val_Accuracy')
plt.title('accuracy')
plt.legend()
plt.show()

# Evaluate model
print('-' * 60)

train_score = model.evaluate(train, verbose= 1)
val_score = model.evaluate(val, verbose= 1)
test_score = model.evaluate(test, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 40)
print("validation Loss: ", val_score[0])
print("validation Accuracy: ", val_score[1])
print('-' * 40)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

# Predictions

preds = model.predict_generator(test)
y_pred = np.argmax(preds, axis=1)


# %%writefile deployment.py

import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.models import load_model


st.title("Brain tumor detection")
st.header('Please upload an image')

file = st.file_uploader('', type = ['png', 'jpg', 'jpeg'])

if file is not None:
  image = Image.open(file).convert('RGB')
  st.image(image, use_column_width = True)

  if st.button('Predict'):
    model = load_model('brain_tumor_model.h5')

    image_tensor = tf.keras.preprocessing.image.img_to_array(image)
    image_tensor = tf.image.resize(image_tensor, (256, 256))
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor /= 255.0

    pred_Y = model.predict(image_tensor)

    if pred_Y[0] > 0.5:
      st.write('There is brain tumor')
    else:
      st.write('There is no brain tumor')


# ! streamlit run deployment.py & npx localtunnel --port 8501

# Save model

# Freeze Model
model.trainable = False
model.summary()

# Save Model
model.save(os.path.join('deployment','brain_tumor_model.h5'))