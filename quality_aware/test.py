#my implement
import pickle
import random
from os.path import join

import array
import ujson as json
import numpy as np
root = join('F://', 'Amazon Dataset', 'Electronics')
def read_asin_set():
    with open(join(root, 'asin_list_Electronics.pickle'), 'rb') as f:
        asin_list=pickle.load(f)
        return set(asin_list)

def read_meta():
    root = join('F://', 'Amazon Dataset', 'Electronics')
    filename = 'fixed_meta_Electronics.json'
    meta = {}
    with open(join(root, filename), 'r') as f:
        data = json.load(f)
        for item in data:
            meta[item['asin']] = 0
    return meta


def read_visual(asin_set):
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield {'asin': asin}
    img_path = join('F://', 'FDMDownload', 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    visual_asin=[]
    try:
        for item in it:
            asin = item['asin'].decode('utf-8')
            if asin in asin_set:
                visual_asin.append(asin)
    except EOFError as e:
        pass
    return set(visual_asin)


def read_text_vec():
    with open(join(root, 'doc_vec_Electronics.pickle'), 'rb') as f:
        tmp = pickle.load(f)
        res = []
        for item in tmp:
            res.append(item[0])
        return set(res)

def read_rating():
    with open(join(root, 'item_rating.pickle'), 'rb') as f:
        return pickle.load(f)


from quality_aware.torch_version.Network import Net
from torch import Tensor
import torch
a=torch.zeros((10,15))
b=torch.zeros((10,11))
c=torch.cat((a,b),dim=1)
print(c.shape)
input()
asin_set=read_asin_set()
print(len(asin_set))
visual_asin=read_visual(asin_set)
print(len(visual_asin))
text_asin = read_text_vec()
print(len(text_asin))
rating = read_rating()
err=0

avaliable_asin=[]

for asin in asin_set:
    if asin not in visual_asin:
        print(asin,'not in visual')
        continue
    if asin not in text_asin:
        print(asin,'not in text')
        continue
    if asin not in rating:
        print(asin,'not in rating')
        continue
    avaliable_asin.append(asin)
avaliable_asin=set(avaliable_asin)
with open(join(root,'avaliable_asin_set.pickle'),'wb')as f:
    pickle.dump(avaliable_asin,f)
print(len(asin_set),err)
print('check ok')
input()
text_feature=read_text_vec()
print(type(text_feature))
input()
meta=read_meta()
for key in meta:
    print(key)
    print(meta[key])
    input()

SubTraingKeys=set(random.sample(asin_set,11000))
Testkeys=asin_set-SubTraingKeys



