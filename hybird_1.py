import numpy as np
import tensorflow as tf
from pre_processing import *
from tensorflow.contrib import rnn

X_test, y_test, X_train, y_train = get_all_data()

x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)

input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")

pool1 = tf.layers.average_pooling2d(inputs=input_layer, pool_size=[1, 20], strides=[1, 20])

conv_2 = tf.layers.conv2d(
    inputs=pool1,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

conv_3 = tf.layers.conv2d(
    inputs=conv_2,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)

pool_4 = tf.layers.max_pooling2d(inputs=conv_3, pool_size=[2, 2], strides=2)

conv_4 = tf.layers.conv2d(
    inputs=pool_4,
    filters=64,
    kernel_size=[11, 1],
    padding="valid",
    activation=tf.nn.relu)

conv_4_flat = tf.reshape(conv_4, [-1, 25, 64])
conv_4_reshape = tf.transpose(conv_4_flat, [0, 2, 1])
x = tf.unstack(conv_4_reshape, 25, axis=2)

lstm_cell = rnn.BasicLSTMCell(100)
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

    for i in range(50):
        inputs = {x_in: X_train, y_real: y_train}
        cost, _ = sess.run([loss, optimizer], feed_dict=inputs)
        print("loss is:", cost)
        inputs = {x_in: X_test, y_real: y_test}
        accu = sess.run(my_acc, feed_dict=inputs)
        print("accuracy is:", accu)
