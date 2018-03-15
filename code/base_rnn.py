import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from code.pre_processing import get_data

X_test, y_test, X_train, y_train = get_data()

# TF input
x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)

x = tf.unstack(x_in, 1000, axis=2)
# TODO: stacking LSTM :https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
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

    for i in range(10):
        inputs = {x_in: X_train, y_real: y_train}
        cost, _ = sess.run([loss, optimizer], feed_dict=inputs)
        print("loss is:", cost)
        inputs = {x_in: X_test, y_real: y_test}
        accu = sess.run(my_acc, feed_dict=inputs)
        print("accuracy is:", accu)
