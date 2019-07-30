import os
import pickle
from collections import defaultdict
from os.path import join
import ujson as json
#import json
'''
def parse(path):
    res=[]
    with open(path,'r')as f:
        for i,line in enumerate(f):
            o=eval(line)
            if 'asin' in o and 'categories' in o and 'related' in o:
                res.append({'asin':         o['asin'],
                            'categories':   o['categories'],
                            'related':      o['related']})

    with open(join('E:/', 'Machine Learning Data Temp', 'related_meta.pickle'),'wb')as f:
        pickle.dump(res,f)
file_path=root = join('E:/', 'Machine Learning Data Temp', 'metadata.json')
parse(file_path)
'''



item_category={}

def parse(path,name):
    global item_category
    sub_set=set()
    com_set=set()
    item_list=[]
    with open(path, 'r')as f:

        for line in f:
            o=eval(line)
            if 'related' not in o:
                continue
            tmp={}
            tmp['asin']=o['asin']
            tmp['related']=o['related']
            item_list.append()
            relate=o['related']
            if 'also_bought' in relate:
                com_set|=(set([(o['asin'], r_item) for r_item in relate['also_bought']]))
            if 'bought_together' in relate:
                com_set|=(set([(o['asin'], r_item) for r_item in relate['bought_together']]))
            if 'also_viewed' in relate:
                sub_set|=(set([(o['asin'], r_item) for r_item in relate['also_viewed']]))
            if 'buy_after_viewing' in relate:
                sub_set|=(set([(o['asin'], r_item) for r_item in relate['buy_after_viewing']]))
    ins=len(sub_set&com_set)
    all=len(sub_set|com_set)
    print(name,':\t',ins,'/',all,'\t','{:.2f}%'.format(ins/all*100))
    #input()

root=join('E:/', 'Machine Learning Data Temp','meta')
files = os.listdir(root)
for f in files:
    if os.path.isfile(join(root,f)):
        name=f.split('_',maxsplit=1)[-1].split('.')[0]
        parse(join(root,f),name)


#parse(join('E:/', 'Machine Learning Data Temp', 'meta_Electronics.json'))
