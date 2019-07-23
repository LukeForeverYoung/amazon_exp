import array


import argparse
import pickle
from os.path import join
import array
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import AffinityPropagation

root = join('F://', 'Amazon Dataset', 'Electronics')

def read_visual(key_set):
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            if asin.decode('utf-8') not in key_set:
                continue

            feature = np.asarray(a.tolist())
            feature = feature.reshape((-1))
            feature=feature.astype(np.float16)
            #feature=np.zeros((1,4096))
            yield {'asin': asin, 'feature': feature}

    img_path = join('F://', 'FDMDownload', 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    # debug
    # it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
    imagefeature = []
    asin_label=[]
    try:
        for item in it:

            asin = item['asin'].decode('utf-8')
            imagefeature.append(item['feature'])
            asin_label.append(asin)
    except EOFError as e:
        pass
    imagefeature=np.asarray(imagefeature)
    return imagefeature,asin_label


def read_data_available():
    with open(join(root, 'available_asin_set.pickle'), 'rb')as f:
        return pickle.load(f)

available=read_data_available()
feature,label=read_visual(available)
print('read ok',len(feature))
kd=KDTree(feature)
print('fit ok')
del feature

with open(join(root,'ap_param.pickle'),'wb') as f:
    pickle.dump(kd,f)
print('save')