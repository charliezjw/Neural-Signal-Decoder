import numpy as np
import tensorflow as tf
from pre_processing import get_data

X_test, y_test, X_train, y_train = get_data()

x_in = tf.placeholder(tf.float32, [None, 22, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
train_mode = tf.placeholder(tf.bool)
y_temp = tf.one_hot(y_real, 4)
input_layer = tf.reshape(x_in, [-1, 22, 1000, 1], name="reshaped_x")

conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv1_norm = tf.contrib.layers.batch_norm(inputs=pool1, is_training=train_mode)
conv1_drop = tf.layers.dropout(inputs=conv1_norm, rate=0.5, training=train_mode)

conv2 = tf.layers.conv2d(
    inputs=conv1_drop,
    filters=64,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
conv2_norm = tf.contrib.layers.batch_norm(inputs=pool2, is_training=train_mode)
conv2_drop = tf.layers.dropout(inputs=conv2_norm, rate=0.5, training=train_mode)

pool2_flat = tf.reshape(conv2_drop, [-1, (5*250*64)])
aff3 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=aff3, units=4)

# calculate accuracy
prediction = tf.argmax(logits, 1, output_type=tf.int32)
my_acc = tf.reduce_mean(tf.cast(tf.equal(y_real, prediction), tf.float32))
test_accu_summ = tf.summary.scalar("test_accu", my_acc)
train_accu_summ = tf.summary.scalar("train_accu", my_acc)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_temp, logits=logits))
test_loss_sum = tf.summary.scalar("test_loss", loss)
train_loss_sum = tf.summary.scalar("train_loss", loss)

optimizer = tf.train.AdamOptimizer().minimize(loss)


writer = tf.summary.FileWriter("./cnn_2/1")
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    writer.add_graph(sess.graph)

    for k in range(15):
        t_size = np.array(range(X_train.shape[0]))
        np.random.shuffle(t_size)
        test_mask = t_size[:100]
        for i in range(10):
            inputs = {x_in: X_train[test_mask], y_real: y_train[test_mask], train_mode: True}
            cost, _, summ_1, summ_2 = sess.run([loss, optimizer, train_accu_summ, train_loss_sum], feed_dict=inputs)
            writer.add_summary(summ_1, i + k*15)
            writer.add_summary(summ_2, i + k*15)

            print("k is:", k)
            inputs = {x_in: X_test, y_real: y_test, train_mode: False}
            accu, cost, summ_1, summ_2 = sess.run([my_acc, loss, test_accu_summ, test_loss_sum], feed_dict=inputs)
            print("test accuracy is:", accu)
            writer.add_summary(summ_1, i + k*15)
            writer.add_summary(summ_2, i + k*15)
