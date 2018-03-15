import numpy as np
import tensorflow as tf
from code.pre_processing import get_data

X_test, y_test, X_train, y_train = get_data()

print(X_test.shape)
print(y_test.shape)
print(X_train.shape)
print(y_train.shape)

x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)
input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")

conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

pool2_flat = tf.reshape(pool2, [-1, (5*250*64)])
aff3 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=aff3, units=4)

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

