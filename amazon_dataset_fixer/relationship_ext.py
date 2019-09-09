
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
    bought_together_dic={}
    also_viewed_dic={}
    buy_after_viewing_dic={}
    substitutes=[]
    compliments=[]
    for item in meta_data:
        if item['asin'] not in asin_set:
            continue
        relate=item['related']
        also_bought=[]
        bought_together=[]
        also_viewed = []
        buy_after_viewing = []
        if 'also_bought' in relate:
            compliments.extend([(item['asin'], r_item) for r_item in relate['also_bought'] if r_item in asin_set])
            also_bought.extend([r_item for r_item in relate['also_bought'] if r_item in asin_set])
        if 'bought_together' in relate:
            compliments.extend([(item['asin'], r_item) for r_item in relate['bought_together'] if r_item in asin_set])
            bought_together.extend([r_item for r_item in relate['bought_together'] if r_item in asin_set])
        if 'also_viewed' in relate:
            substitutes.extend([(item['asin'],r_item) for r_item in relate['also_viewed'] if r_item in asin_set])
            also_viewed.extend([r_item for r_item in relate['also_viewed'] if r_item in asin_set])
        if 'buy_after_viewing' in relate:
            substitutes.extend([(item['asin'], r_item) for r_item in relate['buy_after_viewing'] if r_item in asin_set])
            buy_after_viewing.extend([r_item for r_item in relate['buy_after_viewing'] if r_item in asin_set])
        if len(also_bought)!=0:
            also_bought_dic[item['asin']]=set(also_bought)
        if len(bought_together)!=0:
            bought_together_dic[item['asin']]=set(bought_together)
        if len(also_viewed)!=0:
            also_viewed_dic[item['asin']]=set(also_viewed)
        if len(buy_after_viewing)!=0:
            buy_after_viewing_dic[item['asin']]=set(buy_after_viewing)
    relationship['also_bought']=also_bought_dic
    relationship['bought_together']=bought_together_dic
    relationship['also_viewed']=also_viewed_dic
    relationship['buy_after_viewing']=buy_after_viewing_dic
    if save:
        with open(join(root,'item_relation.pickle'),'wb') as f:
            pickle.dump(relationship, f)
        with open(join(root,'substitutes.pickle'),'wb') as f:
            pickle.dump(substitutes,f)
        with open(join(root,'compliments.pickle'),'wb')as f:
            pickle.dump(compliments,f)
    return relationship


def read_available_asin():
    with open(join(root, 'available_asin_set.pickle'), 'rb')as f:
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
