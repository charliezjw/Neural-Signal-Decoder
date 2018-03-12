import numpy as np
import tensorflow as tf
import h5py
from tensorflow.contrib import rnn

A01T = h5py.File("../project_datasets/A01T_slice.mat", 'r')
X = np.copy(A01T["image"])
y = np.copy(A01T["type"])
y = y[0, 0:X.shape[0]:1]
y = np.asarray(y, dtype=np.int32)

X = X[:, :22, :]
# np.savetxt('test.txt', X, fmt='%5s', delimiter=',')

for i in range(len(y)):
    if y[i] == 769:
        y[i] = 0
    elif y[i] == 770:
        y[i] = 1
    elif y[i] == 771:
        y[i] = 2
    elif y[i] == 772:
        y[i] = 3
    else:
        y[i] = 0
X = X.reshape(-1, 22*1000)
y = y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]
X = X.reshape(-1, 22, 1000)


print(X.shape)

# TF input
x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)

x = tf.unstack(x_in, 1000, axis=2)

lstm_cell = rnn.BasicLSTMCell(100)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

logits = tf.layers.dense(inputs=outputs[-1], units=4)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_temp, logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(1):
        inputs = {x_in: X, y_real: y}
        cost, _, tt = sess.run([loss, optimizer, logits], feed_dict=inputs)
        print("loss is:", cost)
        # print(tt)