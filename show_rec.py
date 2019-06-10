import mxnet as mx
import mxnet.ndarray as nd
from skimage import io
import numpy as np
from mxnet import recordio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

path_imgidx = '/workspace/dataset/faces_webface_112x112/train.idx'
path_imgrec = '/workspace/dataset/faces_webface_112x112/train.rec'

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

for i in tqdm(range(501195)):
    header, s = recordio.unpack(imgrec.read_idx(i+1))
    img = mx.image.imdecode(s).asnumpy()
    label =str(header.label)
    id = str(i)

    plt.imshow(img)
    plt.title('id=' + str(i) + 'label=' + str(header.label))
    plt.pause(0.1)
    print('id=' + str(i) + 'label=' + str(header.label))
