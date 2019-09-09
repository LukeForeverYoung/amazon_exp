import itertools
import json
import pickle
from os.path import join
#import ujson as json
import re
root = join('F://', 'Amazon Dataset', 'Electronics')

'''
读meta-data主要是为抽取text_feature或者其他未知任务做准备,其读入原作者的json,修正为正规json格式并存储
'''
def reader():
    root=join('F://','Amazon Dataset','Electronics')
    filename='meta_Electronics.json'
    out_filename='fixed_meta_Electronics.pickle'
    obj_list=[]
    asin_list=[]
    with open(join(root,filename),'r') as f:
        err=0
        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=eval(tmp)
            obj_list.append(tmp)
            asin_list.append(tmp['asin'])
        print(err)
    print(len(obj_list))
    with open(join(root,out_filename),'wb') as f:
        pickle.dump(obj_list,f)
    with open(join(root, 'asin_list_Electronics.pickle'), 'wb') as f:
        pickle.dump(set(asin_list), f)


reader()