# MobileFaceNet_keras

keras implementation for MobileFaceNet

## TODO

- [ ] use more keras structure form to adjust code.
- [ ] add summary callbacks.(effect visualization)
- [ ] add fintune code.
- [ ] complete comment.

## 准备数据集

1. choose one of The following links to download dataset which is provide by insightface. (Special Recommend MS1M-refine-v2)
* [MS1M-refine-v2@BaiduDrive](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ), [MS1M-refine-v2@GoogleDrive](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
* [Refined-MS1M@BaiduDrive](https://pan.baidu.com/s/1nxmSCch), [Refined-MS1M@GoogleDrive](https://drive.google.com/file/d/1XRdCt3xOw7B3saw0xUSzLRub_HI4Jbk3/view)
* [VGGFace2@BaiduDrive](https://pan.baidu.com/s/1c3KeLzy), [VGGFace2@GoogleDrive](https://www.dropbox.com/s/m9pm1it7vsw3gj0/faces_vgg2_112x112.zip?dl=0)
* [Insightface Dataset Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
2. move dataset to ${MobileFaceNet_TF_ROOT}/datasets.
3. run ${MobileFaceNet_TF_ROOT}/utils/data_process.py.

## 效果

|  size  | LFW(%) | Val@1e-3(%) | inference@xxx(ms) |
| ------ | ------ | ----------- | --------------------- |
|  4.8M  | 99.1+ |        |                   |

## 训练

## 微调

## 测试

### h5 compare log

```text
Images:
0: /workspace/dataset/face_ms1m1/1.0/Figure_118.png
1: /workspace/dataset/face_ms1m1/1.0/Figure_119.png
2: /workspace/dataset/face_ms1m1/2.0/Figure_163.png
3: /workspace/dataset/face_ms1m1/2.0/Figure_164.png
4: /workspace/dataset/face_ms1m1/3.0/Figure_217.png
5: /workspace/dataset/face_ms1m1/3.0/Figure_218.png
6: /workspace/dataset/face_ms1m1/4.0/Figure_227.png
7: /workspace/dataset/face_ms1m1/4.0/Figure_228.png

Distance matrix
        0         1         2         3         4         5         6         7     
0    0.0000    0.5812    1.8437    1.9863    1.8857    2.2854    1.7215    1.4050  
1    0.5812    0.0000    1.7019    1.5926    1.6813    2.2266    1.7883    1.6246  
2    1.8437    1.7019    0.0000    0.5321    1.8039    1.6855    1.8162    1.4440  
3    1.9863    1.5926    0.5321    0.0000    1.7004    1.8784    1.7943    1.6289  
4    1.8857    1.6813    1.8039    1.7004    0.0000    0.5434    1.5834    1.6485  
5    2.2854    2.2266    1.6855    1.8784    0.5434    0.0000    1.7389    1.7090  
6    1.7215    1.7883    1.8162    1.7943    1.5834    1.7389    0.0000    0.6836  
7    1.4050    1.6246    1.4440    1.6289    1.6485    1.7090    0.6836    0.0000  
```

### tflite compare log

```text

INPUTS: 
[{'name': 'input_1', 'index': 160, 'shape': array([  1, 112, 112,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
OUTPUTS: 
[{'name': 'lambda_1/embeddings', 'index': 161, 'shape': array([  1, 128], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
Images:
0: /workspace/dataset/face_ms1m1/1.0/Figure_118.png
1: /workspace/dataset/face_ms1m1/1.0/Figure_119.png
2: /workspace/dataset/face_ms1m1/2.0/Figure_163.png
3: /workspace/dataset/face_ms1m1/2.0/Figure_164.png
4: /workspace/dataset/face_ms1m1/3.0/Figure_217.png
5: /workspace/dataset/face_ms1m1/3.0/Figure_218.png
6: /workspace/dataset/face_ms1m1/4.0/Figure_227.png
7: /workspace/dataset/face_ms1m1/4.0/Figure_228.png

Distance matrix
        0         1         2         3         4         5         6         7     
0    0.0000    0.5812    1.8437    1.9863    1.8857    2.2854    1.7215    1.4050  
1    0.5812    0.0000    1.7019    1.5926    1.6813    2.2266    1.7883    1.6246  
2    1.8437    1.7019    0.0000    0.5321    1.8039    1.6855    1.8162    1.4440  
3    1.9863    1.5926    0.5321    0.0000    1.7004    1.8784    1.7943    1.6289  
4    1.8857    1.6813    1.8039    1.7004    0.0000    0.5434    1.5834    1.6485  
5    2.2854    2.2266    1.6855    1.8784    0.5434    0.0000    1.7389    1.7090  
6    1.7215    1.7883    1.8162    1.7943    1.5834    1.7389    0.0000    0.6836  
7    1.4050    1.6246    1.4440    1.6289    1.6485    1.7090    0.6836    0.0000  
```

## 使用

### 加载h5

```python
import keras
from nets.net import prelu
images = (images - 127.5) * 0.0078125
model = keras.models.load_model(args.model, custom_objects={'prelu': prelu})
emb = model.predict(images)
# 距离
dist = np.sum(np.square(np.subtract(emb[i, :], emb[j, :])))
```

## 参考

1. [facenet](https://github.com/davidsandberg/facenet)
2. [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)