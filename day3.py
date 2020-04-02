
# %% [markdown]
# Image Processing & Model Training 📷


# %%
#  imports
from keras.layers import Dense, Flatten
from keras import Sequential
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
print("tensorflow version = {}".format(tf.__version__))
print("keras version={}".format(keras.__version__))
np.set_printoptions(linewidth=5000)


# %%
# Load images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)


# %%
# print actual values to console
print(x_train[0])


# %%
# matplotlib-prettyprint and corresponding label
plt.imshow(x_train[50])
print(y_train[50])


# %%
# normalize the data
x_train = x_train/255.0
x_test = x_test/255.0


# %%
# now specify the model you want.
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(units=256, activation='relu'))
# output layer
model.add(Dense(units=10, activation='softmax'))

print(model.summary())


# %%
print(x_train.shape)


# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# %%
model.fit(x_train, y_train, epochs=20)


# %%
