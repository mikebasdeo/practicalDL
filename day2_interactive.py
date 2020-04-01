
# %% [markdown]
# ### Linear Regression Practice
# ### Mike Basdeo

# %%
# imports
import matplotlib.pyplot as mpl
import tensorflow as tf
print("tensorflow version = ", tf.__version__)


# %%
#  data creation (X and Y)
def data_creation(w=0.1, b=0.5, n=100):
    # x = actual
    X = tf.random.uniform(shape=(n,))
    noise = tf.random.normal(shape=(n,), mean=0.0, stddev=0.01)
    #  y = prediction
    Y = X*w + b + noise
    return X.numpy(), Y.numpy()


X, Y = data_creation(n=100)
print('data created')
print("X , Y = ", type(X), type(Y))


# %%
print('histogram of the actual values')
mpl.hist(X)
mpl.show()
print('scatter plot of the actual vs predictions')
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
# create two helper functions (predict, and meanSquared)
def predict(x, w=w, b=b):
    y = w*x + b
    return y


def mean_squared_error(y_pred, Y):
    return tf.reduce_mean(tf.square((y_pred-Y)))


print('created helper functions')
# %%
print(mean_squared_error(predict(X), Y))


# %%
w_guess = 0.1
b_guess = 0.5
print(mean_squared_error(predict(X), Y))


# %%

w = tf.Variable(0.0)
b = tf.Variable(-1.0)
learning_rate = 0.1
steps = 200

for step in range(steps):

    # tensorflow steps in to handle derivative calculations
    with tf.GradientTape() as tape:
        predictions = predict(X, w=w, b=b)
        loss = mean_squared_error(predictions, Y)

    gradients = tape.gradient(loss, [w, b])

    w.assign_sub(gradients[0]*learning_rate)
    b.assign_sub(gradients[1]*learning_rate)

    if step % 20 == 0:
        print("Step {}".format(step))


# %%
w

# %%
b

# %%
# Final
w_true = 0.1
b_true = 0.5
w_guess = w
b_guess = b
mpl.plot(X, Y, 'b.')
mpl.plot([0, 1], [0*w_true+b_true, 1*w_true+b_true], 'g:')
mpl.plot([0, 1], [0*w_guess+b_guess, 1*w_guess+b_guess], 'r:')


# %%
