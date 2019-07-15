
import pickle
from collections import defaultdict
from os.path import join
import ujson as json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
root = join('F://', 'Amazon Dataset', 'Electronics')
def make_also_bought(meta_data,asin_set,save=False):
    relationship={}
    also_bought_dic={}
    bought_toghter_dic={}
    for item in meta_data:
        if item['asin'] not in asin_set:
            continue
        relate=item['related']
        also_bought=[]
        bought_together=[]
        if 'also_bought' in relate:
            also_bought.extend([r_item for r_item in relate['also_bought'] if r_item in asin_set])
        if 'bought_together' in relate:
            bought_together.extend([r_item for r_item in relate['bought_together'] if r_item in asin_set])
        if len(also_bought)!=0:
            also_bought_dic[item['asin']]=set(also_bought)
        if len(bought_together)!=0:
            bought_toghter_dic[item['asin']]=set(bought_together)
    relationship['also_bought']=also_bought_dic
    relationship['bought_together']=bought_toghter_dic
    if save:
        with open(join(root,'item_relation.pickle'),'wb') as f:
            pickle.dump(relationship, f)
    return relationship


def read_avaliable_asin():
    with open(join(root, 'avaliable_asin_set.pickle'), 'rb')as f:
        return pickle.load(f)


def read_meta():
    filename = 'fixed_meta_Electronics.json'
    with open(join(root, filename), 'r') as f:
        data=json.load(f)
    return data

meta_data=read_meta()
asin_set=read_avaliable_asin()
als=make_also_bought(meta_data,asin_set,save=True)

print(als.keys())
print(len(als['also_bought'].keys()))
print(len(als['bought_together'].keys()))
