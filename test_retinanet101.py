import tensorflow as tf
from scipy import misc

import numpy as np
import sys
import utils.visualization as vis
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
import json

# Load Resnet-50 network pretrained on imagenet.

# # Create network to handle new class.
# wisnet = Wisnet(resnet)

# logits = wisnet.output
#
# # Prepare new class data
#
# #  Compute loss
# labels = tf.placeholder(tf.float32, [None, gconf['class_num']])
# tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
# loss = tf.losses.get_total_loss()
# tf.summary.scalar('loss', loss)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=gconf['learning_rate'])
# train_op = optimizer.minimize(loss)
#
# correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels, axis=1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.summary.scalar('train_acc', accuracy)
#
# summary = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    saver = tf.train.import_meta_graph('/home/autel/PycharmProjects/onlineWisdom/models/resnet-nhwc-2018-02-07/model.ckpt-112603.meta')

    tb_log_writer = tf.summary.FileWriter('logs', sess.graph)