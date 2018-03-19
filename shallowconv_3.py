import numpy as np
import tensorflow as tf
from pre_processing import *

X_test, y_test, X_train, y_train = get_all_data()
X_test, y_test, X_train, y_train = crop_data(X_test, y_test, X_train, y_train)

# apply cropped training
x_in = tf.placeholder(tf.float32, [None, 22, 500], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_test_in = tf.placeholder(tf.int32, [None], name="test_y")
train_mode = tf.placeholder(tf.bool)
y_temp = tf.one_hot(y_real, 4)

input_layer = tf.reshape(x_in, [-1, 22, 500, 1], name="reshaped_x")

conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=40,
    kernel_size=[1, 25],
    padding="valid",
    activation=None)

conv1_norm = tf.contrib.layers.batch_norm(inputs=conv1, is_training=train_mode)
conv1_drop = tf.layers.dropout(inputs=conv1_norm, rate=0.5, training=train_mode)

conv2 = tf.layers.conv2d(
    inputs=conv1_drop,
    filters=40,
    kernel_size=[22, 1],
    padding="valid",
    activation=tf.nn.elu)

conv2_norm = tf.contrib.layers.batch_norm(inputs=conv2, is_training=train_mode)
conv2_drop = tf.layers.dropout(inputs=conv2_norm, rate=0.5, training=train_mode)

pool1 = tf.layers.average_pooling2d(inputs=conv2_drop, pool_size=[1, 75], strides=[1, 15])

out_flat = tf.reshape(pool1, [-1, (1*27*40)])

aff1 = tf.layers.dense(inputs=out_flat, units=512, activation=tf.nn.relu)
aff1_norm = tf.contrib.layers.batch_norm(inputs=aff1, is_training=train_mode)
aff1_drop = tf.layers.dropout(inputs=aff1_norm, rate=0.5, training=train_mode)

logits = tf.layers.dense(inputs=out_flat, units=4, activation=tf.nn.relu)

# calculate accuracy
prediction = tf.argmax(logits, 1, output_type=tf.int32)
my_acc = tf.reduce_mean(tf.cast(tf.equal(y_real, prediction), tf.float32))
train_accu_summ = tf.summary.scalar("train_accu", my_acc)

# the test run
prediction_test = tf.argmax(logits, 1, output_type=tf.int32)
temp_1 = tf.reshape(prediction_test, [-1, 450])
temp_2 = tf.transpose(temp_1)
b = tf.argmax(tf.bincount(temp_2[0]), output_type=tf.int32)
b = tf.stack([b, tf.argmax(tf.bincount(temp_2[1]), output_type=tf.int32)], 0)
for i in range(2, 450):
    max_indx = tf.argmax(tf.bincount(temp_2[i]), output_type=tf.int32)
    b = tf.concat([b, [max_indx]], 0)

my_acc_test = tf.reduce_mean(tf.cast(tf.equal(y_test_in, b), tf.float32))
test_accu_summ = tf.summary.scalar("test_accu", my_acc_test)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_temp, logits=logits))
train_loss_sum = tf.summary.scalar("train_loss", loss)

optimizer = tf.train.AdamOptimizer().minimize(loss)


writer = tf.summary.FileWriter("./shallowconv_3/2")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    writer.add_graph(sess.graph)

    for k in range(300):
        t_size = np.array(range(X_train.shape[0]))
        np.random.shuffle(t_size)
        test_mask = t_size[:1]
        # for i in range(15):s
        inputs = {x_in: X_train[test_mask], y_real: y_train[test_mask], train_mode: True, y_test_in: [1]}
        cost, _, summ_1, summ_2 = sess.run([loss, optimizer, train_accu_summ, train_loss_sum], feed_dict=inputs)
        writer.add_summary(summ_1, k)
        writer.add_summary(summ_2, k)

        print("k is:", k)
        print(X_test.shape)
        inputs = {x_in: X_test, y_real: [1], y_test_in: y_test, train_mode: False}
        accu, summ_1 = sess.run([my_acc_test, test_accu_summ], feed_dict=inputs)
        print("test accuracy is:", accu)
        writer.add_summary(summ_1, k)
