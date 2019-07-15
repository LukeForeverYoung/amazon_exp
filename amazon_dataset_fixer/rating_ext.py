import pickle

import array
from os.path import join
import numpy as np
import ujson as json
from collections import defaultdict

root = join('F://', 'Amazon Dataset', 'Electronics')
def read_ratings():
    rating_dic=defaultdict(list)
    i=0
    with open(join(root,'reviews_Electronics.json'),'r')as f:
        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=json.loads(f.readline())

            rating=tmp['overall']
            # 二值化,3为作者给定超参数
            if rating>3:
                rating=1
            else:
                rating=0

            if 'overall' in tmp:
                rating_dic[tmp['asin']].append(rating)
            i+=1
            if i%100000==0:
                print(i)
    return rating_dic

def cal_theta(r_list):
    r_list=np.asarray(r_list)
    length=r_list.shape[0]
    distence=length+1-np.sum(r_list)
    distence/=(length+2)
    return distence

rating_dic=read_ratings()
result={}
for item in rating_dic:
    rating_list=rating_dic[item]
    result[item]=cal_theta(rating_list)

with open(join(root, 'item_rating.pickle'), 'wb') as f:
    pickle.dump(result, f)
