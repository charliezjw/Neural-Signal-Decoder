import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from pre_processing import *

X_test, y_test, X_train, y_train = get_all_data()


# TF input
x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)

input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")
pool1 = tf.layers.average_pooling2d(inputs=input_layer, pool_size=[1, 10], strides=[1, 10])
pool1_reshape = tf.reshape(pool1, [-1, 22, 100])

x = tf.unstack(pool1_reshape, 100, axis=2)
# TODO: stacking LSTM :https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
lstm_cell = rnn.BasicLSTMCell(200, activation=tf.nn.relu)
outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

logits = tf.layers.dense(inputs=outputs[-1], units=4)

# calculate accuracy
prediction = tf.argmax(logits, 1, output_type=tf.int32)
my_acc = tf.reduce_mean(tf.cast(tf.equal(y_real, prediction), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_temp, logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    for k in range(15):
        t_size = np.array(range(X_train.shape[0]))
        np.random.shuffle(t_size)
        test_mask = t_size[:400]

        for i in range(106):
            inputs = {x_in: X_train[test_mask], y_real: y_train[test_mask]}
            cost, _ = sess.run([loss, optimizer], feed_dict=inputs)
            print("training loss is:", cost)
            inputs = {x_in: X_test, y_real: y_test}
            accu, cost = sess.run([my_acc, loss], feed_dict=inputs)
            print("----test accuracy is:", accu)
            print("test cost is:", cost)
