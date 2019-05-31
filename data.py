import mxnet as mx

def read_rec(idx_path, rec_path):
    idx_path = os.path.expanduser('~/workspace/MobileFaceNet_TF_i/datasets/faces_webface_112x112/train.idx')
    rec_path = os.path.expanduser('~/workspace/MobileFaceNet_TF_i/datasets/faces_webface_112x112/train.rec')
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')

    s = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(s)
    # HEADER(flag=2, label=array([490624., 501196.], dtype=float32), id=0, id2=0)
    imgidx = list(range(1, int(header.label[0])))

    # 图片数量
    pic_number = int(header.label[0]) - 1
    # 类别数量
    class_number = int(header.label[1]) - int(header.label[0])

    pics = []

    for i, index in enumerate(imgidx):
        img_info = imgrec.read_idx(index)
        header, img_raw = mx.recordio.unpack(img_info)
        label = int(header.label)

        pics.append([label, img_raw])

    print(len(pics)) # 490623
    
    return pics

