import tensorflow as tf
import matplotlib.pyplot as mpl

# myVar = tf.random.uniform([1])
# print(myVar.numpy())

# create a list of size 100
# with values between 0 and 1.


def data_creation(w=0.1, b=0.5, n=100):

    X = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(n,), mean=0.0, stddev=0.01)
    Y = X*w + b + noise
    return X.numpy(), Y.numpy()


X, Y = data_creation(n=100)
# mpl.hist(X)
# mpl.show()


mpl.plot(X, Y, 'bo')
mpl.show()
