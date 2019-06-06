import tensorflow as tf
import numpy as np
import math
import tensorflow.keras as keras
from tensorflow.keras.callbacks import *
import os
from pathlib import Path
import cv2
import tensorflow.keras.backend as K
from tqdm import tqdm

from utils.data_process import parse_function, load_data
# from losses.face_losses import arcface_loss
# from nets.MobileFaceNet import inference
from verification import evaluate
from scipy.optimize import brentq
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import numpy as np
import time
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout,PReLU,Layer
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape,DepthwiseConv2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Constant
from tensorflow.keras import regularizers

from tensorflow.keras.regularizers import l2

from tensorflow.keras import backend as K
import math

weight_decay = 5e-5  # l2正则化decay常量


batch_norm_params = {
    'center': True,
    'scale': True,
    'momentum': 0.995,
    'epsilon': 2e-5,
}

MODEL_FILE = 'MobileFaceNet.h5'
LITE_FILE  = 'MobileFaceNet.tflite'

NUM_PICTURES=490623
NUM_CLASSES=10572
BATCH_SIZE=90
TARGET_SIZE=(112,112)
TFRECORD_PATH='/workspace/dataset/faces_webface_112x112/tfrecords/tran.tfrecords'

train_batch_size = 90
test_batch_size = 100
eval_datasets = ['lfw']
eval_db_path = '/workspace/dataset/faces_webface_112x112/'
eval_nrof_folds = 10
tfrecords_file_path = '/workspace/dataset/faces_webface_112x112/tfrecords/'
summary_path = '/workspace/output/summary'
ckpt_path = '/workspace/output/ckpt'
pretrained_model = False
log_file_path = '/workspace/output/logs'
ckpt_best_path = '/workspace/output/ckpt_best'
saver_maxkeep = 50
summary_interval = 400
ckpt_interval = 200
validate_interval = 500
show_info_interval = 50
log_device_mapping = False
log_histograms = False
prelogits_norm_loss_factor = 2e-5
prelogits_norm_p = 1.0
max_epoch = 12
image_size = [112, 112]
embedding_size = 128

# prepare validate datasets
# ver_list = []
# ver_name_list = []
# for db in eval_datasets:
#     print('begin db %s convert.' % db)
#     data_set = load_data(db, image_size, eval_db_path)
#     ver_list.append(data_set)
#     ver_name_list.append(db)

# # output file path
# if not os.path.exists(log_file_path):
#     os.makedirs(log_file_path)
# if not os.path.exists(ckpt_best_path):
#     os.makedirs(ckpt_best_path)
# if not os.path.exists(ckpt_path):
#     os.makedirs(ckpt_path)
# # create log dir
# subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
# log_dir = os.path.join(os.path.expanduser(log_file_path), subdir)
# if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
#     os.makedirs(log_dir)


def my_generator(tfrecord_path=TFRECORD_PATH,batch_size=BATCH_SIZE,out_num=NUM_CLASSES):
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

        #img = tf.py_func(random_rotate_image, [img], tf.uint8)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.subtract(img, 127.5)
        img = tf.multiply(img,  0.0078125)
        img = tf.image.random_flip_left_right(img)
        label = tf.cast(features['label'], tf.int64)
#         label = tf.one_hot(label,out_num)
#         label = tf.reshape(label,(-1,))
#         one_hot = tf.one_hot(label,out_num)
        return (img, label)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
#     sess = K.get_session()
    # training datasets api config
    tfrecords_f = os.path.join(tfrecord_path)
    dataset = tf.data.TFRecordDataset(tfrecords_f)
    dataset = dataset.map(parse_function)
#     dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element, sess
    # begin iteration
#     while(True):
#         sess.run(iterator.initializer)
#         while True:
#             try:
#                 images, labels = sess.run(next_element)
# #                 for i in range(len(images)):
# #                     images[i,...] = cv2.cvtColor(images[i, ...], cv2.COLOR_RGB2BGR)
#                 yield images,labels
#             except tf.errors.OutOfRangeError:
# #                 print("End of dataset")
#                 break


def my_generator_wrapper():
    for image,label in my_generator():
        yield ([image,label],label)
#         yield image,label
        
def test_generator():
    a = my_generator_wrapper()
    for i in tqdm(range(50000)):
        a.__next__()
        
def learning_rate_schedule(epoch, lr, boundaries, values):
    """
    # Argument:
        epoch: now epoch
        lr: learning rate to schedule
        boundaries: Number of epochs for learning rate piecewise.
        values: target value of learning rate
    """
    if epoch <= boundaries[0]:
        t = values[0]
    for low, high, v in zip(boundaries[:-1],boundaries[1:],values[1:]):
        if low < epoch <= high:
            t = v
    if epoch > boundaries[-1]:
        t = values[-1]
        
    K.get_session().run(lr.assign(t))
    return epoch,t

class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
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



def flow_wrapper(flow):
    """自定义wrapper，将(x,y)变成([x,y],y)"""
    while True:
        x,y = flow.next()
        yield ([x,y],y)

def prelu(input, name=''):
    """自定义prelu"""
    alphas = K.variable(K.constant(0.25,dtype=tf.float32,shape=[input.get_shape()[-1]]),name=name + 'prelu_alphas')
    pos = K.relu(input)
    neg = alphas * (input - K.abs(input)) * 0.5
    return pos + neg

cval = Constant(0.25)  # prelu α 初始常量

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides, kernel_initializer='glorot_normal',kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(axis=channel_axis,**batch_norm_params)(x)
    x = PReLU(cval)(x)
#     x = Activation(relu)(x)
    return x


def _bottleneck(inputs, filters, kernel, t, s, r=False):
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

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = PReLU(cval)(x)
#     x = Activation(relu)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer='glorot_normal',kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis,**batch_norm_params)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
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

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


class ArcFace(Layer):
    """改进的softmax，得出的结果再与真是结果之间求交叉熵"""
    def __init__(self, n_classes=10, s=64.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        print(input_shape)
        print('build')
        super(ArcFace, self).build(input_shape[0])
        print('build2')
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1].value, self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)
        print('build3')

#     def call(self, inputs):
#         x, y = inputs # x为embeddings，y为labels
#         c = K.shape(x)[-1]  # 特征维度
#         # 1. normalize feature
#         x = tf.nn.l2_normalize(x, axis=1)
#         # 2. normalize weights
#         W = tf.nn.l2_normalize(self.W, axis=0)
#         # dot product
#         # 全连接层，x的结构为（None，128）w的结构为（128，n_classes）。logits的结构为(None,n_classes)
#         # (np.random.randn(5,128) @ np.random.randn(128,10)).shape # (5, 10)
#         # 3. 计算xW得到预测向量y
#         logits = x @ W
#         # add margin
#         # clip logits to prevent zero division when backward
#         theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
#         target_logits = tf.cos(theta + self.m)
#         # sin = tf.sqrt(1 - logits**2)
#         # cos_m = tf.cos(logits)
#         # sin_m = tf.sin(logits)
#         # target_logits = logits * cos_m - sin * sin_m
#         logits = logits * (1 - y) + target_logits * y
#         # feature re-scale
#         # 9. 对所有值乘上固定值s
#         logits *= self.s
#         out = tf.nn.softmax(logits)
#         print(out)
#         return out

    
    def call(self, inputs):
        embedding, labels = inputs
        labels = tf.reshape(labels,shape=(-1,))
        print('labels:',labels)
        
        out_num = self.n_classes
        w_init=None
        s=64.
        m=0.5
        
        cos_m = tf.cos(m)
        sin_m = tf.sin(m)
        mm = sin_m * m  # issue 1
        threshold = tf.cos(math.pi - m)
        with tf.compat.v1.variable_scope('arcface_loss'):
            # inputs and weights norm
            embedding_norm = tf.norm(tensor=embedding, axis=1, keepdims=True)
            embedding = tf.compat.v1.div(embedding, embedding_norm, name='norm_embedding')
#             weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
#                                       initializer=w_init, dtype=tf.float32)
            weights = self.W
            weights_norm = tf.norm(tensor=weights, axis=0, keepdims=True)
            weights = tf.compat.v1.div(weights, weights_norm, name='norm_weights')
            # cos(theta+m)
            cos_t = tf.matmul(embedding, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)
            print('labels:',labels,'out_num',out_num)
            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
#             mask = tf.reshape(mask,(-1,))
#             mask = labels
            print(mask)
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')
            print(inv_mask)
            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
            print(s_cos_t)
            logit = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
            print(logit)
#             inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
            inference_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
            print(inference_loss)
#             inference_loss = tf.nn.softmax(logit)
            print(inference_loss)
        return inference_loss
    
    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def MobileFaceNets(input_shape=(112,112,3), n_classes=10, k=128):
    """MobileFaceNets"""
    inputs = Input(shape=input_shape) #112x112，(img-127.5)/255
    y      = Input(shape=(1,), dtype=tf.int32)
#     y      = Input(shape=(n_classes,))
    x = _conv_block(inputs, 64, (3, 3), strides=(2, 2))
    
    # depthwise conv3x3
    x = DepthwiseConv2D(3, strides=(1, 1), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU(cval)(x)
#     x = Activation(relu)(x)
    
    # 5层bottleneck
    x = _inverted_residual_block(x, 64, (3, 3), t=2, strides=2, n=5)
    x = _inverted_residual_block(x, 128, (3, 3), t=4, strides=2, n=1)
    x = _inverted_residual_block(x, 128, (3, 3), t=2, strides=1, n=6)
    x = _inverted_residual_block(x, 128, (3, 3), t=4, strides=2, n=1)
    x = _inverted_residual_block(x, 128, (3, 3), t=2, strides=1, n=2)
    
    # conv1x1
    x = _conv_block(x, 512, (1, 1), strides=(1, 1))
    
    # linear GDConv7x7
    x = DepthwiseConv2D(7, strides=(1, 1), depth_multiplier=1, padding='valid')(x)
#     x = Dropout(0.3, name='Dropout')(x)
    
    x = Conv2D(k, (1, 1), padding='same',kernel_initializer='glorot_normal',kernel_regularizer=l2(1e-10))(x)
    
    
#     x = Activation(keras.activations.re)
    x = Reshape((k,))(x)
    print(x)
#     x = keras.layers.Lambda(lambda o: K.l2_normalize(o, axis=1))(x)
    epsilon=1e-10
    x = keras.layers.Lambda(lambda o: o/K.sqrt(K.maximum(K.sum(K.square(o),axis=1,keepdims=True),epsilon)))(x)
#     x = tf.nn.l2_normalize(x, 1, 1e-10, name='embeddings')
    
    # x 为embeddings， y为embeddings对应的类别标签，output为
    output = ArcFace(n_classes=n_classes, regularizer=None)([x, y])
    
    model = Model([inputs, y], output)
#     plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
    print(model.input,model.output)
    return model


def my_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

K.clear_session()

model = MobileFaceNets(n_classes=NUM_CLASSES)
model.compile(optimizer=keras.optimizers.Adam(0.1,beta_1=0.9, beta_2=0.999, epsilon=0.1),
      loss=my_loss,
      metrics=['accuracy'])

# EMAer = ExponentialMovingAverage(model) # 在模型compile之后执行
# EMAer.inject() # 在模型compile之后执行

# model.compile(optimizer=keras.optimizers.sgd(0.1),loss=my_loss,metrics=['accuracy'])
val_model = keras.models.Model(inputs=model.inputs[0], outputs=model.layers[-3].output)

train_batch_size = 90
test_batch_size = 100
eval_datasets = ['lfw']
eval_db_path = '/workspace/dataset/faces_webface_112x112/'
eval_nrof_folds = 10
tfrecords_file_path = '/workspace/dataset/faces_webface_112x112/tfrecords/'
summary_path = '/workspace/output/summary'
ckpt_path = '/workspace/output/ckpt'
pretrained_model = False
log_file_path = '/workspace/output/logs'
ckpt_best_path = '/workspace/output/ckpt_best'
saver_maxkeep = 50
summary_interval = 400
ckpt_interval = 200
validate_interval = 500
show_info_interval = 50
log_device_mapping = False
log_histograms = False
prelogits_norm_loss_factor = 2e-5
prelogits_norm_p = 1.0
max_epoch = 12
image_size = [112, 112]
embedding_size = 128

lr_schedule = [4, 7, 9, 11]
values=[0.1, 0.01, 0.001, 0.0001, 0.00001]

# prepare validate datasets
ver_list = []
ver_name_list = []
for db in eval_datasets:
    print('begin db %s convert.' % db)
    data_set = load_data(db, image_size, eval_db_path)
    ver_list.append(data_set)
    ver_name_list.append(db)

# output file path
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
if not os.path.exists(ckpt_best_path):
    os.makedirs(ckpt_best_path)
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
# create log dir
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
log_dir = os.path.join(os.path.expanduser(log_file_path), subdir)
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
    os.makedirs(log_dir)
    
# g = my_generator_wrapper()
iterator, next_element, g_sess = my_generator()

# epoch = -1
count = 0
total_accuracy = {}
for i in range(max_epoch):
    # 调整学习率
    _, lr = learning_rate_schedule(i,model.optimizer.lr,lr_schedule,values)
    print('epoch:{}, lr:{}'.format(i, lr))
    # 初始化迭代器
    g_sess.run(iterator.initializer)
    
    while True:
        try:
            
            x, y = g_sess.run(next_element)
            
            images_train,labels_train = ([x,y],y)

            start = time.time()
#             _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
#             sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op, Accuracy_Op],
#                      feed_dict=feed_dict)
            loss, accuracy = model.train_on_batch(images_train, labels_train)
    
            end = time.time()
            pre_sec = train_batch_size/(end - start)

            count += 1
            # print training information
            if count > 0 and count % show_info_interval == 0:
#                 print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, reg_loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
#                       (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val), acc_val, pre_sec))
                print('epoch %d, total_step %d, loss is %.6f, training accuracy is %.6f, time %.3f samples/sec' %
                      (i, count, loss, accuracy, pre_sec))

            # save summary
#             if count > 0 and count % summary_interval == 0:
#                 feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
#                 summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
#                 summary.add_summary(summary_op_val, count)

            # save ckpt files
            if count > 0 and count % ckpt_interval == 0:
#                 filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.ckpt'
#                 EMAer.apply_ema_weights()
                filename = 'MobileFaceNet_iter_{:d}'.format(count) + '.h5'
                filename = os.path.join(ckpt_path, filename)
                val_model.save(filename)
                
#                 EMAer.reset_old_weights()

            # validate
            if count > 0 and count % validate_interval == 0:
                print('\nIteration', count, 'testing...')
                
#                 EMAer.apply_ema_weights()
                
                for db_index in range(len(ver_list)):
                    start_time = time.time()
                    data_sets, issame_list = ver_list[db_index]
                    emb_array = np.zeros((data_sets.shape[0], embedding_size))
                    nrof_batches = data_sets.shape[0] // test_batch_size
                    for index in range(nrof_batches): # actual is same multiply 2, test data total
                        start_index = index * test_batch_size
                        end_index = min((index + 1) * test_batch_size, data_sets.shape[0])

#                         feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
#                         emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                        emb_array[start_index:end_index, :] = val_model.predict(data_sets[start_index:end_index, ...])

                    tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=eval_nrof_folds)
                    duration = time.time() - start_time

                    print("total time %.3fs to evaluate %d images of %s" % (duration, data_sets.shape[0], ver_name_list[db_index]))
                    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
                    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                    print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                    auc = metrics.auc(fpr, tpr)
                    print('Area Under Curve (AUC): %1.3f' % auc)
#                     eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
#                     print('Equal Error Rate (EER): %1.3f\n' % eer)

                    with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[db_index])), 'at') as f:
                        f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                    if ver_name_list == 'lfw' and np.mean(accuracy) > 0.992:
                        print('best accuracy is %.5f' % np.mean(accuracy))
                        filename = 'MobileFaceNet_iter_best_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(ckpt_best_path, filename)
                        saver.save(sess, filename)
                        
#                 EMAer.reset_old_weights()
                
        except tf.errors.OutOfRangeError:
            print("End of epoch %d" % i)
            break
