
# %% [markdown]
# ### Linear Regression Practice


# %%
# imports
import matplotlib.pyplot as mpl
import tensorflow as tf
print('imports loaded')


# %%
#  data creation (X and Y)
def data_creation(w=0.1, b=0.5, n=100):
    X = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(n,), mean=0.0, stddev=0.01)
    Y = X*w + b + noise
    return X.numpy(), Y.numpy()


X, Y = data_creation(n=100)
print(type(X))


# %%
# raw output data
mpl.hist(X)
mpl.show()


# %%
#  plot all the predictions
mpl.plot(X, Y, 'b.')
mpl.show()


# %%
# goal line
w = 0.1
b = 0.5
mpl.plot(X, Y, 'b.')
mpl.plot([0, 1], [0*w+b, 1*w+b], 'g:')


# %%
#  plot initial guess (red line)
w_guess = 0
b_guess = 0
mpl.plot(X, Y, 'b.')
mpl.plot([0, 1], [0*w+b, 1*w+b], 'g:')
mpl.plot([0, 1], [0*w_guess+b_guess, 1*w_guess+b_guess], 'r:')


# %%
def predict(x):
    y = w_guess*x + b_guess
    return y


# %%
def mean_squared_error(y_pred, Y):

    return tf.reduce_mean(tf.square((y_pred-Y)))


# %%
print(mean_squared_error(predict(X), Y))


# %%
w_guess = 0.1
b_guess = 0.5


# %%
