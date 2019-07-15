import pickle

import array
from os.path import join
import numpy as np


def read_visual(asin_set):
    '''
    图像特征无需预处理,此函数在使用visual_feature时帮助读入
    从预处理特征集中读取特征,将特征向量转化为numpy形式(节约内存)
    :param asin_set: 无残缺
    :return:
    '''
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            if asin not in asin_set:
                continue
            feature = np.asarray(a.tolist())
            feature = feature.reshape((feature.shape[0], 1))
            yield {'asin': asin, 'feature': feature}

    img_path = join('F://', 'FDMDownload', 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    # debug
    # it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
    imagefeature = {}
    try:
        for item in it:
            asin = item['asin'].decode('utf-8')
            if asin in asin_set:
                imagefeature[asin] = item['feature']
    except EOFError as e:
        pass
    return imagefeature
