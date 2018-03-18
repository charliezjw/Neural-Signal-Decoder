import numpy as np
import tensorflow as tf
from pre_processing import *
from tensorflow.contrib import rnn

X_test, y_test, X_train, y_train = get_all_data()

x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")

conv_1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[22, 1],
    padding="same",
    activation=tf.nn.relu)

pool1 = tf.layers.average_pooling2d(inputs=conv_1, pool_size=[1, 10], strides=[1, 10])

conv_3 = tf.layers.conv2d(
    inputs=pool1,
    filters=1,
    kernel_size=[1, 1],
    padding="same",
    )

conv_4_flat = tf.reshape(conv_3, [-1, 22, 100])
print(conv_4_flat.shape)
x = tf.unstack(conv_4_flat, 100, axis=2)

lstm_cell = rnn.BasicLSTMCell(100)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
outputs, final_state = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)

fc_5 = tf.layers.dense(inputs=outputs[-1], units=512, activation=tf.nn.relu, kernel_regularizer=regularizer)
fc_6 = tf.layers.dense(inputs=outputs[-1], units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)

logits = tf.layers.dense(inputs=fc_6, units=4)

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
            print("test accuracy is:", accu)
            print("test cost is:", cost)
