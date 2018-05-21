"""
Script to train a new class to pretrained ResNeXt-101 model.
"""
import tensorflow as tf
from nets import resnet_v2
from scipy import misc

import numpy as np
import sys
import utils.visualization as vis
import matplotlib.pyplot as plt
from tensorflow.contrib import slim
import json
from preprocessing import inception_preprocessing as preprocess


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_conf():
    conf = {}
    conf['input_size'] = 299
    conf['class_num'] = 1001
    conf['is_training'] = True
    conf['weight_decay'] = 0.0005

    conf['checkpoint'] = '/home/autel/libs/modelzoo/tensorflow-models/resnet_v2_50/resnet_v2_50.ckpt'
    conf['log_dir'] = 'logs'

    return conf


def trans_name(labels):
    """
    Translate labels to names.
    :param labels: 1-D ndarray containing label number.
    :return: list of string.
    """
    label_name = open('imagenet1000_clsid_to_human.json', 'r')
    label_name = json.load(label_name)
    return [label_name[str(l)] for l in labels]


def main(_):
    gconf = load_conf()
    tf.reset_default_graph()

    # Load base network.
    input_imgs = tf.placeholder(tf.float32, [None, None, 3])
    img_input = preprocess.preprocess_for_eval(input_imgs, gconf['input_size'], gconf['input_size'], central_fraction=0.9)
    img_input = tf.expand_dims(img_input, axis=0)
    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=gconf['weight_decay'],
                                                   use_batch_norm=True)):
        resnet, _ = resnet_v2.resnet_v2_50(inputs=img_input,
                                        num_classes=gconf['class_num'],
                                        is_training=gconf['is_training'])

    probs = tf.nn.softmax(resnet)
    # labels = tf.argmax(probs, 1)
    probs, labels = tf.nn.top_k(probs, 5)

    # Prepare data.
    img = misc.imread('/home/autel/data/exp_imgs/beagle.jpg')

    # Load Resnet-50 network pretrained on imagenet.
    saver = tf.train.Saver()

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

    step_cnt = 0
    with tf.Session() as sess:
        tb_log_writer = tf.summary.FileWriter(gconf['log_dir'], sess.graph)

        sess.run(init_op)
        saver.restore(sess, gconf['checkpoint'])

        probs_val, labels_val = sess.run([probs, labels], feed_dict={input_imgs:img})
        print(labels_val)

        for i in range(len(probs_val)):
            labels_name = trans_name(labels_val[i] - 1)
            for label_name, prob in zip(labels_name, probs_val[i]):
                print('label: %s, prob: %f' % (label_name, prob))

        # vis.visulizeClassByName(img, labels_name[0], probs_val[0][labels_val[0]], True)
        # plt.waitforbuttonpress()

        # tb_log_writer.add_summary()

        # for _ in range(gconf['epoch_num']):
        #     sess.run(dataset_itr.initializer)
        #
        #     while True:
        #         step_cnt = step_cnt + 1
        #         try:
        #             # train
        #             imgs_input, labels_input = sess.run([img_batch, labels_batch])
        #
        #
        #             # lab_pred, lab_batch = sess.run([labels_pred, class_id_batch], feed_dict={input_imgs:imgs_input,
        #             #                                                       labels: labels_input});
        #             # print(lab_pred, lab_batch)
        #             # exit(0)
        #             #
        #             # for img, class_onehot in zip(imgs_input, labels_input):
        #             #     utils.visulizeClass(img, class_onehot, class_dict, hold=True)
        #             #     plt.waitforbuttonpress()
        #
        #             summary_val, loss_val, train_acc, _ = sess.run([summary, loss, accuracy, train_op], feed_dict={input_imgs:imgs_input,
        #                                                                   labels: labels_input})
        #
        #             if step_cnt % gconf['log_step'] == 0:
        #                 tb_log_writer.add_summary(summary_val, step_cnt)
        #                 print('Step %d, loss: %f, train_acc: %f'%(step_cnt, loss_val, train_acc))
        #         except tf.errors.OutOfRangeError:
        #             # log statistics
        #             # break
        #             break


if __name__ == '__main__':
    tf.app.run()