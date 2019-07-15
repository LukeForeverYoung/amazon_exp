
import pickle
from collections import defaultdict
from os.path import join
import ujson as json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
root = join('F://', 'Amazon Dataset', 'Electronics')
def ext_relationship(meta_data, asin_set, save=False):
    '''
    抽取关系
    :param meta_data: 元数据,包含related信息
    :param asin_set: 数据完整商品asin集合
    :param save: 选择是否保存
    :return: 返回关系集合{
                            'also_bought':{'asin':set()},
                            'bought_together':{'asin':set()}
                        }
    '''
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


def read_available_asin():
    with open(join(root, 'avaliable_asin_set.pickle'), 'rb')as f:
        return pickle.load(f)


def read_meta():
    filename = 'fixed_meta_Electronics.json'
    with open(join(root, filename), 'r') as f:
        data=json.load(f)
    return data

meta_data=read_meta()
asin_set=read_available_asin()
relationship=ext_relationship(meta_data, asin_set, save=True)

print(relationship.keys())
print(len(relationship['also_bought'].keys()))
print(len(relationship['bought_together'].keys()))
