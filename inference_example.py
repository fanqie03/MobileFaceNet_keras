import keras
import tensorflow as tf
import scipy.misc as misc
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('tflite_file', help='file of tflite model')
    parser.add_argument('image_files', type=str, nargs='+', help='file of image which is aligned')
    parser.add_argument('--image_size', default=112, type=int, help='size of image to feed the model')

    return parser.parse_args()

def load_images(args):
    # imgs = np.empty((len(args.image_files), args.image_size, args.image_size, 3), np.float32)
    for i, image in enumerate(args.image_files):
        # img = cv2.imread(image)
        # img = np.asarray(Image.open(image))
        img = misc.imread(image)
        img = misc.imresize(img, (args.image_size, args.image_size))
        img = (img - 127.5) * 0.0078125
        img = img.astype(np.float32)
        # imgs[i, ...] = img
    return np.stack([img])

def _main(args):
    interpreter = tf.lite.Interpreter(model_path=args.tflite_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    images = load_images(args)
    interpreter.set_tensor(input_details[0]['index'], images)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])
    print('='*30)
    print('input is :')
    print(images,images.shape)
    print('=' * 30)
    print('output is :')
    print(emb, emb.shape)



if __name__ == '__main__':
    """
    python inference_example.py arch/pretrained_model/MobileFaceNet_iter_566000.tflite data/Figure_110.png
    """
    _main(get_args())
