import numpy as np
import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)

sum_op = tf.add(x1, x2)

with tf.Session() as session:
    sum_result = session.run(sum_op, feed_dict={x1: 2.0, x2: 0.5})

print(sum_result)
