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
|  4.8M  | 99.1+  |     97.0    |                   |

## 训练

1. 运行[data_process.py](utils/data_process.py)将insightface中的预处理数据集转为`tfrecode`格式
2. 运行[train_nets.py](train_nets.py)训练出一个.h5模型
3. 运行[convert_tflite.py](convert_tflite.py)将.h5格式转为.tflite格式

## 微调

TODO

## 测试

TODO

### h5 compare log

```text
Images:
0: data/1.0/Figure_110.png
1: data/1.0/Figure_111.png
2: data/2.0/Figure_129.png
3: data/2.0/Figure_130.png
4: data/3.0/Figure_212.png
5: data/3.0/Figure_213.png
6: data/4.0/Figure_227.png
7: data/4.0/Figure_228.png
8: data/cmf/1.png
9: data/cmf/2.png
10: data/zyy/1.png
11: data/zyy/2.png

Distance matrix
        0         1         2         3         4         5         6         7         8         9         10         11     
0    0.0000    0.4479    1.9588    2.0532    2.3128    2.2733    1.6796    1.4539    1.3723    1.3951    1.2134    1.3029  
1    0.4479    0.0000    2.2099    2.2770    2.3163    2.4148    1.7598    1.7522    1.6421    1.5925    1.4884    1.4468  
2    1.9588    2.2099    0.0000    0.3399    2.0805    1.7233    1.8289    1.8432    1.2553    1.1766    1.6389    1.5797  
3    2.0532    2.2770    0.3399    0.0000    1.8132    1.4081    1.4626    1.5342    1.2111    1.1683    1.8445    1.8163  
4    2.3128    2.3163    2.0805    1.8132    0.0000    0.3596    1.9348    1.9537    2.0169    1.9201    2.1521    2.1166  
5    2.2733    2.4148    1.7233    1.4081    0.3596    0.0000    1.7134    1.6631    1.7724    1.7743    2.2111    2.0889  
6    1.6796    1.7598    1.8289    1.4626    1.9348    1.7134    0.0000    0.5197    1.7797    1.6080    1.3491    1.4088  
7    1.4539    1.7522    1.8432    1.5342    1.9537    1.6631    0.5197    0.0000    1.7442    1.5961    1.4316    1.4927  
8    1.3723    1.6421    1.2553    1.2111    2.0169    1.7724    1.7797    1.7442    0.0000    0.3919    1.2670    1.4540  
9    1.3951    1.5925    1.1766    1.1683    1.9201    1.7743    1.6080    1.5961    0.3919    0.0000    1.2849    1.4578  
10    1.2134    1.4884    1.6389    1.8445    2.1521    2.2111    1.3491    1.4316    1.2670    1.2849    0.0000    0.1708  
11    1.3029    1.4468    1.5797    1.8163    2.1166    2.0889    1.4088    1.4927    1.4540    1.4578    0.1708    0.0000  
```

### tflite compare log

```text

INPUTS: 
[{'name': 'input_1', 'index': 160, 'shape': array([  1, 112, 112,   3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
OUTPUTS: 
[{'name': 'lambda_1/embeddings', 'index': 161, 'shape': array([  1, 128], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
Images:
0: data/1.0/Figure_110.png
1: data/1.0/Figure_111.png
2: data/2.0/Figure_129.png
3: data/2.0/Figure_130.png
4: data/3.0/Figure_212.png
5: data/3.0/Figure_213.png
6: data/4.0/Figure_227.png
7: data/4.0/Figure_228.png
8: data/cmf/1.png
9: data/cmf/2.png
10: data/zyy/1.png
11: data/zyy/2.png

Distance matrix
        0         1         2         3         4         5         6         7         8         9         10         11     
0    0.0000    0.4479    1.9588    2.0532    2.3128    2.2733    1.6796    1.4539    1.3723    1.3951    1.2134    1.3029  
1    0.4479    0.0000    2.2099    2.2770    2.3163    2.4148    1.7598    1.7522    1.6421    1.5925    1.4884    1.4468  
2    1.9588    2.2099    0.0000    0.3399    2.0805    1.7233    1.8289    1.8432    1.2553    1.1766    1.6389    1.5797  
3    2.0532    2.2770    0.3399    0.0000    1.8132    1.4081    1.4626    1.5342    1.2111    1.1683    1.8445    1.8163  
4    2.3128    2.3163    2.0805    1.8132    0.0000    0.3596    1.9348    1.9537    2.0169    1.9201    2.1521    2.1166  
5    2.2733    2.4148    1.7233    1.4081    0.3596    0.0000    1.7134    1.6631    1.7724    1.7743    2.2111    2.0889  
6    1.6796    1.7598    1.8289    1.4626    1.9348    1.7134    0.0000    0.5197    1.7797    1.6080    1.3491    1.4088  
7    1.4539    1.7522    1.8432    1.5342    1.9537    1.6631    0.5197    0.0000    1.7442    1.5961    1.4316    1.4927  
8    1.3723    1.6421    1.2553    1.2111    2.0169    1.7724    1.7797    1.7442    0.0000    0.3919    1.2670    1.4540  
9    1.3951    1.5925    1.1766    1.1683    1.9201    1.7743    1.6080    1.5961    0.3919    0.0000    1.2849    1.4578  
10    1.2134    1.4884    1.6389    1.8445    2.1521    2.2111    1.3491    1.4316    1.2670    1.2849    0.0000    0.1708  
11    1.3029    1.4468    1.5797    1.8163    2.1166    2.0889    1.4088    1.4927    1.4540    1.4578    0.1708    0.0000  
  
```

## 使用

more detail in [compare.py](compare.py)

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