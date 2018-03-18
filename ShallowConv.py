import numpy as np
import tensorflow as tf
from pre_processing import get_data

X_test, y_test, X_train, y_train = get_data()

x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)

input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")

conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=40,
    kernel_size=[25, 1],
    padding="same",
    activation=tf.nn.relu)

conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=40,
    kernel_size=[22, 1],
    padding="valid",
    activation=tf.nn.relu)

pool1 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[1, 50], strides=[1, 50])

out_flat = tf.reshape(pool1, [-1, (1*20*40)])
aff1 = tf.layers.dense(inputs=out_flat, units=512, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=aff1, units=4, activation=tf.nn.relu)

# calculate accuracy
prediction = tf.argmax(logits, 1, output_type=tf.int32)
my_acc = tf.reduce_mean(tf.cast(tf.equal(y_real, prediction), tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_temp, logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())

    for i in range(15):
        inputs = {x_in: X_train, y_real: y_train}
        cost, _ = sess.run([loss, optimizer], feed_dict=inputs)
        print("loss is:", cost)
        inputs = {x_in: X_test, y_real: y_test}
        accu = sess.run(my_acc, feed_dict=inputs)
        print("accuracy is:", accu)
