# Copyright 2018 The AI boy xsr-ai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileFaceNets.

MobileFaceNets, which use less than 1 million parameters and are specifically tailored for high-accuracy real-time
face verification on mobile and embedded devices.

here is MobileFaceNets architecture, reference from MobileNet_V2 (https://github.com/xsr-ai/MobileNetv2_TF).

As described in https://arxiv.org/abs/1804.07573.

  MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification on Mobile Devices

  Sheng Chen, Yang Liu, Xiang Gao, Zhen Han

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, PReLU, Layer
from keras.layers import Activation, BatchNormalization, add, Reshape, DepthwiseConv2D
from keras.utils import plot_model
from keras.activations import relu
from keras.initializers import Constant
from keras import regularizers
import keras

from keras.regularizers import l2

from keras import backend as K

from collections import namedtuple
import functools


# Conv and InvResBlock namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# InvResBlock defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer


class prelu(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(prelu, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alphas = self.add_weight('alpha',shape=(input_shape[-1],),dtype=tf.float32,initializer=Constant(0.25))
        super(prelu, self).build(input_shape)

    def call(self, x, **kwargs):
        pos = tf.nn.relu(x)
        neg = self.alphas * (x - tf.abs(x)) * 0.5
        return pos + neg

    # def prelu(self, input, name=''):
    #     with tf.variable_scope(name, reuse=False):
    #         alphas = tf.get_variable(name='prelu_alphas',
    #                                  initializer=tf.constant(0.25, dtype=tf.float32, shape=[input.get_shape()[-1]]))
    #         pos = tf.nn.relu(input)
    #         neg = alphas * (input - tf.abs(input)) * 0.5
    #         keras.layers.Lambda()
    #         return pos + neg

class MobileFaceNet(object):
    Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'ratio'])
    DepthwiseConv = namedtuple('DepthwiseConv', ['kernel', 'stride', 'depth', 'ratio'])
    InvResBlock = namedtuple('InvResBlock', ['kernel', 'stride', 'depth', 'ratio', 'repeate'])

    _CONV_DEFS = [
        Conv(kernel=[3, 3], stride=2, depth=64, ratio=1),
        DepthwiseConv(kernel=[3, 3], stride=1, depth=64, ratio=1),

        InvResBlock(kernel=[3, 3], stride=2, depth=64, ratio=2, repeate=5),
        InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
        InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=6),
        InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
        InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=2),

        Conv(kernel=[1, 1], stride=1, depth=512, ratio=1),
    ]

    def __init__(self):
        # self._Conv_count = 0
        # self._SeparableConv2d_count = 0
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.weight_decay = 5e-5  # l2正则化decay常量

        self.batch_norm_params = {
            'center': True,
            'scale': True,
            'momentum': 0.995,
            'epsilon': 2e-5,
            'axis': self.channel_axis,
            # 'name': 'BatchNorm',
        }

        self.separable_conv2d_params = {
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'padding': 'same',
            'use_bias': False,
            'depthwise_initializer': 'glorot_normal',
            'depthwise_regularizer': None,
        }

        self.cval = Constant(0.25)

        # _CONV_DEFS specifies the MobileNet body

    def calc_count(self, name):
        """
        使命名次数增加
        :param name:
        :return:
        """
        key = '_{}_count'.format(name)
        dic = self.__dict__

        if dic.get(key, 0) == 0:
            name = name
            dic[key] = 0
        else:
            name = '{}_{}'.format(name, dic[key])
            dic[key] += 1
        return name





    def _conv(self, x, filters, kernel_size, strides, padding='same', use_bias=False,
              kernel_initializer='glorot_normal', activation=True, kernel_regularizer=None):
        """

        :param x:
        :param filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :param use_bias:
        :param kernel_initializer:
        :return:
        """
        name = self.calc_count('Conv')
        with tf.name_scope(name):
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                       kernel_initializer=kernel_initializer, name=None, kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization(**self.batch_norm_params)(x)
            if activation:
                x = prelu()(x)
        return x

    def _separable_conv2d(self, x, kernel_size,
                          strides=(1, 1),
                          padding='same',
                          use_bias=False,
                          depthwise_initializer='glorot_normal',
                          depthwise_regularizer=None,
                          activation=True,
                          kernel_regularizer=None):
        """

        :param x:
        :return:
        """
        name = self.calc_count('SeparableConv2d')
        with tf.name_scope(name):
            x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                padding=padding, use_bias=use_bias,
                                depthwise_initializer=depthwise_initializer,
                                depthwise_regularizer=depthwise_regularizer,
                                kernel_regularizer=kernel_regularizer)(x)
            x = BatchNormalization(**self.batch_norm_params)(x)
            if activation:
                x = prelu()(x)
            # print(x)
        return x

    def _bottleneck(self, inputs, filters, kernel, t, s, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            r: Boolean, Whether to use the residuals.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        tchannel = K.int_shape(inputs)[channel_axis] * t

        x = self._conv(inputs, tchannel, (1, 1), (1, 1))
        x = self._separable_conv2d(x, kernel_size=kernel, strides=(s, s))

        x = self._conv(x, filters=filters, kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=l2(self.weight_decay),
                       activation=False)

        if r:
            x = add([x, inputs])
        return x

    def _inverted_residual_block(self, inputs, filters, kernel, t, strides, n):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        x = self._bottleneck(inputs, filters, kernel, t, strides)

        for i in range(1, n):
            x = self._bottleneck(x, filters, kernel, t, 1, True)

        return x

    def inference(self, x, k):
        """

        :param x:
        :return:
        """
        with tf.name_scope('MobileFaceNet'):
            with tf.name_scope('Conv2d_0'):
                x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
                x = BatchNormalization(**self.batch_norm_params)(x)
                x = prelu()(x)
            self._separable_conv2d(x, (3, 3), (1, 1))
            self._conv(x, 64, (1, 1), (1, 1), activation=False)

            # 5层bottleneck
            x = self._inverted_residual_block(x, 64, (3, 3), t=2, strides=2, n=5)
            x = self._inverted_residual_block(x, 128, (3, 3), t=4, strides=2, n=1)
            x = self._inverted_residual_block(x, 128, (3, 3), t=2, strides=1, n=6)
            x = self._inverted_residual_block(x, 128, (3, 3), t=4, strides=2, n=1)
            x = self._inverted_residual_block(x, 128, (3, 3), t=2, strides=1, n=2)

            # conv1x1
            x = self._conv(x, 512, (1, 1), strides=(1, 1))

            with tf.name_scope('Logits'):
                # linear GDConv7x7
                x = self._separable_conv2d(x, kernel_size=(7, 7), strides=(1, 1), padding='valid', activation=False)
                x = self._conv(x, 512, (1, 1), strides=(1, 1), activation=False)
                #     x = Dropout(0.3, name='Dropout')(x)
                with tf.name_scope('LinearConv1x1'):
                    x = self._conv(x, k, (1, 1), strides=(1, 1), kernel_regularizer=l2(1e-10), activation=False)
                    x = Reshape((k,))(x)

        return x

    def test(self, x):
        """

        :param x:
        :return:
        """
        with tf.name_scope('MobileFaceNet'):
            x = self._conv(x, 32, (1, 1), (1, 1), batch_norm_params=self.batch_norm_params)
            x = self._conv(x, 32, (1, 1), (1, 1), batch_norm_params=self.batch_norm_params)
            x = self._conv(x, 32, (1, 1), (1, 1), batch_norm_params=self.batch_norm_params)
            x = self._separable_conv2d(x, (3, 3))
            x = self._separable_conv2d(x, (3, 3))
        return x


if __name__ == '__main__':

    inputs = Input((112, 112, 3))
    m = MobileFaceNet()
    print(m.__dict__)
    x = m.inference(inputs, 512)
    model = Model(inputs=inputs, outputs=x)
    model.compile('adam', keras.losses.categorical_crossentropy)
    # model.summary()
    for v in tf.trainable_variables():
        print(v)
    # MobileFaceNet._CONV_DEFS
