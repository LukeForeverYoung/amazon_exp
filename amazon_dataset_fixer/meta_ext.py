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
    out_filename='fixed_meta_Electronics.json'
    obj_list=[]
    asin_list=[]
    with open(join(root,filename),'r') as f:
        err=0
        while True:
            tmp=f.readline()
            if not tmp:
                break
            try:
                tmp=eval(tmp)
            except Exception as e:
                print(e)
                err+=1
                continue
            if 'imUrl' in tmp and 'title' in tmp and 'description' in tmp and 'related' in tmp:
                n_obj = {}
                n_obj['asin']=tmp['asin']
                n_obj['imUrl'] = tmp['imUrl']
                n_obj['title'] = tmp['title']
                n_obj['description'] = tmp['description']
                n_obj['related'] = tmp['related']
                obj_list.append(n_obj)
                asin_list.append(n_obj['asin'])
        print(err)

    with open(join(root,out_filename),'w') as f:
        json.dump(obj_list,f)
    with open(join(root, 'asin_list_Electronics.pickle'), 'wb') as f:
        pickle.dump(set(asin_list), f)


reader()