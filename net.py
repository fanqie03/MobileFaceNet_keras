from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout,PReLU,Layer
from keras.layers import Activation, BatchNormalization, add, Reshape,DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.activations import relu
from keras.initializers import Constant
from keras import regularizers

from keras import backend as K

weight_decay = 1e-4  # l2正则化decay常量

cval = Constant(0.25)  # prelu α 初始值

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

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
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

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = PReLU(cval)(x)
#     x = Activation(relu)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

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
    
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs # x为embeddings，y为labels
        c = K.shape(x)[-1]  # 特征维度
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        # 全连接层，x的结构为（None，128）w的结构为（128，n_classes）。logits的结构为(None,n_classes)
        # (np.random.randn(5,128) @ np.random.randn(128,10)).shape # (5, 10)
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def MobileFaceNets(input_shape=(112,112,3), n_classes=10, k=128):
    """MobileFaceNets"""
    inputs = Input(shape=input_shape) #112x112，(img-127.5)/255
    y      = Input(shape=(n_classes,))
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
    x = Dropout(0.3, name='Dropout')(x)
    
    
    
    x = Conv2D(k, (1, 1), padding='same')(x)
    x = Reshape((k,))(x)
    # x 为embeddings， y为embeddings对应的类别标签，output为
    output = ArcFace(n_classes=n_classes, regularizer=regularizers.l2(weight_decay))([x, y])
    
    model = Model([inputs, y], output)
#     plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
    print(model.input,model.output)
    return model

