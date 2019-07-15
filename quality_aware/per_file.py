import pickle
from collections import defaultdict
from os.path import join
import ujson as json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

root = join('F://', 'Amazon Dataset', 'Electronics')
def read_meta():
    filename = 'fixed_meta_Electronics.json'
    with open(join(root, filename), 'r') as f:
        data=json.load(f)
    return data

def read_asin_list(meta,save=False):
    asin_list=[]
    for item in meta:
        asin_list.append(item['asin'])
    if save:
        with open(join(root,'asin_list_Electronics.pickle'),'wb') as f:
            pickle.dump(asin_list,f)
    return asin_list

def make_also_bought(meta_data,asin_set,save=False):
    relationship={}
    also_bought_dic={}
    bought_toghter_dic={}
    for item in meta_data:
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
    if save:
        with open(join(root,'item_relation.pickle'),'wb') as f:
            pickle.dump(also_bought_dic, f)
    return also_bought_dic

meta_data=read_meta()
#包含了具有title和description的item
asin_set=set(read_asin_list(meta_data,save=True))
#只保留具有relationship的,不具有的被省略,且对方item也必须在asin_set中


relationship=make_also_bought(meta_data,asin_set,save=True)
'''
also_data=[len(relationship[item]) for item in relationship]
zeor=sum([1 for item in relationship if len(relationship[item])==1])
# visual feature[[asin,[visual_feature]]]

# text feature [[asin,[text feature]]]
'''
