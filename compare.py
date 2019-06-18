"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import scipy.misc as misc
import numpy as np
import sys
import argparse
import keras
from nets.net import prelu
from PIL import Image
import cv2
import os
import re
import tensorflow as tf
from pathlib import Path


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def main(args):
    base_dir = Path(args.base_dir).expanduser()
    image_files = list(base_dir.glob('*/*'))
    setattr(args, 'image_files', image_files)
    images = load_images(args)
    nrof_images = len(args.image_files)
    # Load the model
    if args.model.endswith('.h5'):
        model = keras.models.load_model(args.model, custom_objects={'prelu': prelu})
        emb = model.predict(images)
    elif args.model.endswith('.pb'):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                load_model(args.model)

                # Get input and output tensors, ignore phase_train_placeholder for it have default value.
                inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                emb = sess.run(embeddings, feed_dict={inputs_placeholder: images})
    else:
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print('INPUTS: ')
        print(input_details)
        print('OUTPUTS: ')
        print(output_details)

        # Test model on random input data.
        emb = np.empty(shape=(nrof_images, 128), dtype=np.float32)
        for i, img in enumerate(images):
            img = img.reshape((1, 112, 112, 3))
            interpreter.set_tensor(input_details[0]['index'], img)

            interpreter.invoke()
            emb[i, ...] = interpreter.get_tensor(output_details[0]['index']).reshape((-1,))

    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, args.image_files[i]))
    print('')

    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%1d  ' % i, end='')
        for j in range(nrof_images):
            dist = np.sum(np.square(np.subtract(emb[i, :], emb[j, :])))
            print('  %1.4f  ' % dist, end='')
        print('')


def load_images(args):
    imgs = np.empty((len(args.image_files), args.image_size, args.image_size, 3), np.float32)
    for i, image in enumerate(args.image_files):
        # img = cv2.imread(image)
        # img = np.asarray(Image.open(image))
        img = misc.imread(image)
        img = misc.imresize(img, (args.image_size, args.image_size))
        img = (img - 127.5) * 0.0078125
        imgs[i, ...] = img
    return imgs


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--base_dir', type=str)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
