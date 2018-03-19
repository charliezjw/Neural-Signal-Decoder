import numpy as np
import tensorflow as tf

a = tf.placeholder(tf.int32, [None, 3])

b = tf.convert_to_tensor(tf.argmax(tf.bincount(a[0])))
b = tf.stack([b, tf.argmax(tf.bincount(a[1]))], 0)
for i in range(2, 5):
    max_indx = tf.argmax(tf.bincount(a[i]))
    b = tf.concat([b, [max_indx]], 0)

with tf.Session() as sess:
    t1 = np.asarray([[1, 1, 0], [2, 4, 4], [6, 6, 6], [5, 5, 5], [2, 7, 7]])
    t2, t3 = sess.run([b, max_indx], feed_dict={a: t1})
    print(t2)
    print(t3)
