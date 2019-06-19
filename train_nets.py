# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''

import argparse
import os
import time
from datetime import datetime

import keras
import keras.backend as K
import math
import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.layers import Input, Layer
from keras.models import Model
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics

from nets.net import MobileFaceNet
from utils.common import check_path
from utils.data_process import load_data
from verification import evaluate


def get_parser():
    """
    use demo

    ```shell
    python train_nets_keras.py \
    --class_number 10572 \
    --eval_db_path /workspace/dataset/faces_webface_112x112 \
    --tfrecords_file_path /workspace/dataset/faces_webface_112x112/tfrecords/tran.tfrecords \
    --h5_path /workspace/output/h5 \
    --tflite_path /workspace/tflite \
    --h5_best_path /workspace/output/h5_best \
    ```
    :return:
    """
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=12, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--class_number', required=True, type=int,
                        help='class number depend on your training datasets, MS1M-V1: 85164, MS1M-V2: 85742, webface: 10572')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[4, 7, 9, 11])
    parser.add_argument('--values', default=[0.1, 0.01, 0.001, 0.0001, 0.00001], help='learning rate schedule degree')
    parser.add_argument('--train_batch_size', default=90, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--tfrecords_file_path', default='./datasets/faces_ms1m_112x112/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--h5_path', default='./output/h5', help='the h5 file save path')
    parser.add_argument('--h5_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--tflite_path', default='./output/tflite', help='the tflite file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=50, help='tf.train.Saver max keep h5 files')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--h5_interval', default=2000, type=int, help='intervals to save h5 file')
    parser.add_argument('--validate_interval', default=2000, type=int, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=50, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=2e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)

    args = parser.parse_args()
    return args


def my_generator(tfrecord_path, batch_size):
    """自定义generator

    # Argument
        tfrecord_path:
        batch_size
        out_num: 类别数量，用于生成onehot

    # Return

    """

    def parse_function(example_proto):
        features = {'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(example_proto, features)
        # You can do more image distortion here for training data
        img = tf.image.decode_jpeg(features['image_raw'])
        img = tf.reshape(img, shape=(112, 112, 3))

        img = tf.cast(img, dtype=tf.float32)
        img = tf.subtract(img, 127.5)
        img = tf.multiply(img, 0.0078125)
        img = tf.image.random_flip_left_right(img)
        label = tf.cast(features['label'], tf.int64)
        return (img, label)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # training datasets api config
    tfrecords_f = os.path.join(tfrecord_path)
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element, sess


def learning_rate_schedule(epoch, lr, boundaries, values):
    """
    # Argument:
        epoch: now epoch
        lr: learning rate to schedule
        boundaries: Number of epochs for learning rate piecewise.
        values: target value of learning rate
    """
    t = 1
    if epoch <= boundaries[0]:
        t = values[0]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:]):
        if low < epoch <= high:
            t = v
    if epoch > boundaries[-1]:
        t = values[-1]

    K.get_session().run(lr.assign(t))
    return epoch, t


class ArcFace(Layer):
    """改进的softmax，得出的结果再与真是结果之间求交叉熵"""

    def __init__(self, n_classes=10, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='embedding_weights',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs, **kwargs):
        embedding, labels = inputs
        labels = tf.reshape(labels, shape=(-1,))
        print('labels:', labels)

        out_num = self.n_classes
        w_init = None
        s = 64.
        m = 0.5

        cos_m = tf.cos(m)
        sin_m = tf.sin(m)
        mm = sin_m * m  # issue 1
        threshold = tf.cos(math.pi - m)
        with tf.variable_scope('arcface_loss'):
            # inputs and weights norm
            embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
            embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
            weights = self.W
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')

            cos_t = tf.matmul(embedding, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s * (cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)
            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            inv_mask = tf.subtract(1., mask, name='inverse_mask')
            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
            logit = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
            inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)

        return inference_loss

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """

    def __init__(self, model, momentum=0.999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]

    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)

    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))

    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))

    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


def MobileFaceNets(input_shape=(112, 112, 3), n_classes=10, k=128):
    """MobileFaceNets"""
    inputs = Input(shape=input_shape)  # 112x112，(img-127.5)/128
    y = Input(shape=(1,), dtype=tf.int32)
    m = MobileFaceNet()
    x = m.inference(inputs, k)
    epsilon = 1e-10
    x = keras.layers.Lambda(lambda o: tf.nn.l2_normalize(o, 1, epsilon, name='embeddings'))(x)
    output = ArcFace(n_classes=n_classes, regularizer=None)([x, y])

    model = Model([inputs, y], output)
    print(model.input, model.output)
    return model


def my_loss(y_true, y_pred):
    """只要预测值即可，预测值包含batch个损失，求均值
    """
    return tf.reduce_mean(y_pred)


if __name__ == '__main__':
    with tf.Graph().as_default():
        args = get_parser()

        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args.eval_db_path)
            ver_list.append(data_set)
            ver_name_list.append(db)

        # output file path
        check_path([args.log_file_path,
                    args.h5_best_path,
                    args.h5_path,
                    args.tflite_path,
                    args.summary_path])
        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # g = my_generator_wrapper()
        iterator, next_element, g_sess = my_generator(args.tfrecords_file_path, args.train_batch_size)

        model = MobileFaceNets(n_classes=args.class_number)
        model.compile(optimizer=keras.optimizers.Adam(0.1, beta_1=0.9, beta_2=0.999, epsilon=0.1),
                      loss=my_loss,
                      metrics=['accuracy'])

        # EMAer = ExponentialMovingAverage(model)  # 在模型compile之后执行
        # EMAer.inject()  # 在模型compile之后执行

        val_model = keras.models.Model(inputs=model.inputs[0], outputs=model.layers[-3].output)

        count = 0
        total_accuracy = {}
        for i in range(args.max_epoch):
            # 调整学习率
            _, lr = learning_rate_schedule(i, model.optimizer.lr, args.lr_schedule, args.values)
            print('epoch:{}, lr:{}'.format(i, lr))
            # 初始化迭代器
            g_sess.run(iterator.initializer)

            while True:
                try:

                    x, y = g_sess.run(next_element)

                    images_train, labels_train = ([x, y], y)

                    start = time.time()
                    loss, accuracy = model.train_on_batch(images_train, labels_train)

                    end = time.time()
                    pre_sec = args.train_batch_size / (end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print(
                            'epoch %d, total_step %d, loss is %.6f, training accuracy is %.6f, time %.3f samples/sec' %
                            (i, count, loss, np.mean(accuracy), pre_sec))

                    # save h5 files
                    if count > 0 and count % args.h5_interval == 0:
                        #                 filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
                        # EMAer.apply_ema_weights()
                        filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.h5'
                        filename = os.path.join(args.h5_path, filename)
                        val_model.save(filename)

                        # EMAer.reset_old_weights()

                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')

                        # EMAer.apply_ema_weights()

                        for db_index in range(len(ver_list)):
                            start_time = time.time()
                            data_sets, issame_list = ver_list[db_index]
                            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            nrof_batches = data_sets.shape[0] // args.test_batch_size
                            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                                start_index = index * args.test_batch_size
                                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                emb_array[start_index:end_index, :] = val_model.predict(
                                    data_sets[start_index:end_index, ...])

                            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list,
                                                                             nrof_folds=args.eval_nrof_folds)
                            duration = time.time() - start_time

                            print("total time %.3fs to evaluate %d images of %s" % (
                                duration, data_sets.shape[0], ver_name_list[db_index]))
                            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                            print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                            auc = metrics.auc(fpr, tpr)
                            print('Area Under Curve (AUC): %1.3f' % auc)
                            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                            print('Equal Error Rate (EER): %1.3f\n' % eer)

                            with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])),
                                      'at') as f:
                                f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                            if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                                filename = 'MobileFaceNet_iter_best_{:d}'.format(count) + '.h5'
                                filename = os.path.join(args.h5_best_path, filename)
                                val_model.save(filename)

                        # EMAer.reset_old_weights()

                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
