import itertools
import json
import pickle
from os.path import join
#import ujson as json
import re
root = join('F://', 'Amazon Dataset', 'Electronics')

def reader():
    filename='meta_Electronics.json'

    with open(join(root,filename),'r') as f:
        res={}
        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=eval(tmp)

            if 'categories' in tmp:
                res[tmp['asin']]=tmp['categories']

    return res

res = reader()

with open(join(root, 'item_category.pickle'), 'wb') as f:
    pickle.dump(res, f)