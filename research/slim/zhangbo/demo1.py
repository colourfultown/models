# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午10:31
# @Author  : dabo
# @Site    : 
# @File    : demo1.py
# @Software: PyCharm
import tensorflow as tf

x = tf.constant([[0,33],[3,1],[4,2],[5,8]], dtype=tf.int64)
y = tf.one_hot(x,depth=6)
z = tf.argmax(x, axis=1)
with tf.Session() as sess:
    sess.run(tf.global_variables())
    print(sess.run(z))
pass

tf.metrics.auc()