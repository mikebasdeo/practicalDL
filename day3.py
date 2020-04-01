
# %% [markdown]
# Image Processing & Model Training 📷


# %%
#  imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras import Sequential
print("tensorflow version = {}".format(tf.__version__))
print("keras version={}".format(keras.__version__))


# %%
# Load images
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# %%
np.set_printoptions(linewidth=5000)


# %%
# print actual values to console
print(x_train[0])


# %%
# matplotlib-prettyprint
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

print(model.summary())


# %%
