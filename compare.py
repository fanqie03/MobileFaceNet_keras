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
import tensorflow as tf

def main(args):
    images = load_images(args)
    nrof_images = len(args.image_files)
    # Load the model
    if args.model.endswith('.h5'):
        model = keras.models.load_model(args.model, custom_objects={'prelu': prelu})
        emb = model.predict(images)
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
            img = img.reshape((1,112,112,3))
            interpreter.set_tensor(input_details[0]['index'], img)

            interpreter.invoke()
            emb[i,...] = interpreter.get_tensor(output_details[0]['index']).reshape((-1,))

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
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
