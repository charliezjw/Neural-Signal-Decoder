import numpy as np
import tensorflow as tf
import h5py

A01T = h5py.File("../project_datasets/A01T_slice.mat", 'r')
X = np.copy(A01T["image"])
y = np.copy(A01T["type"])
y = y[0, 0:X.shape[0]:1]
y = np.asarray(y, dtype=np.int32)

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
X = X.reshape(-1, 25*1000)
y = y[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]
X = np.copy(X.reshape(-1, 25, 1000))

x_in = tf.placeholder(tf.float32, [None, 25, 1000], name="input_x")
y_real = tf.placeholder(tf.int32, [None], name="real_y")
y_temp = tf.one_hot(y_real, 4)
input_layer = tf.reshape(x_in, [-1, 25, 1000, 1], name="reshaped_x")

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

pool2_flat = tf.reshape(pool2, [-1, (24*1000*64)//16])
aff3 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

logits = tf.layers.dense(inputs=aff3, units=4)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y_temp, logits=logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(15):
        inputs = {x_in: X, y_real: y}
        cost, _, tt = sess.run([loss, optimizer, logits], feed_dict=inputs)
        print("loss is:", cost)
        # print(tt)
