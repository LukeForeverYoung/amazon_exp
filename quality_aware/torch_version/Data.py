import pickle

import array
from torch.utils.data import Dataset
import torch
from os.path import join
from random import sample, shuffle, randrange
import numpy as np
root = join('F://', 'Amazon Dataset', 'Electronics')
def collate_fn(batch):
    batch_size=len(batch)
    visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label = zip(*batch)
    visual_tensor_1=torch.stack(visual_tensor_1)
    text_tensor_1=torch.stack(text_tensor_1)
    visual_tensor_2 = torch.stack(visual_tensor_2)
    text_tensor_2=torch.stack(text_tensor_2)
    rating_2=torch.stack(rating_2).reshape((batch_size,1))
    label=torch.stack(label).reshape((-1,1))

    return visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label

def collate_with_cat_fn(batch):
    batch_size=len(batch)
    visual_tensor_1,text_tensor_1,category_tensor_1,visual_tensor_2,text_tensor_2,category_tensor_2,rating_2,label = zip(*batch)
    visual_tensor_1=torch.stack(visual_tensor_1)
    text_tensor_1=torch.stack(text_tensor_1)
    category_tensor_1=torch.stack(category_tensor_1)
    visual_tensor_2 = torch.stack(visual_tensor_2)
    text_tensor_2=torch.stack(text_tensor_2)
    category_tensor_2=torch.stack(category_tensor_2)
    rating_2=torch.stack(rating_2).reshape((batch_size,1))
    label=torch.stack(label).reshape((-1,1))
    return visual_tensor_1,text_tensor_1,category_tensor_1,visual_tensor_2,text_tensor_2,category_tensor_2,rating_2,label

class ExtDataset(Dataset):
    def __init__(self,data_list,text,visual,rating,category=None):
        self.data_list=data_list
        self.text=text
        self.visual=visual
        self.rating=rating
        self.category=category
        self.len=len(data_list)

    def __getitem__(self, index):
        tup=self.data_list[index]
        item_key=tup[0]
        target_key=tup[1]

        visual_tensor_1=torch.from_numpy(self.visual[item_key].reshape((-1))).float()
        text_tensor_1=torch.from_numpy(self.text[item_key].reshape((-1))).float()
        visual_tensor_2 = torch.from_numpy(self.visual[target_key].reshape((-1))).float()
        text_tensor_2 = torch.from_numpy(self.text[target_key].reshape((-1))).float()
        rating_2=torch.tensor(self.rating[target_key]).float()
        label = torch.tensor(tup[2]).byte()
        if self.category is not None:
            category_tensor_1=torch.from_numpy(self.category[item_key].reshape((-1))).float()
            category_tensor_2=torch.from_numpy(self.category[target_key].reshape((-1))).float()
            return visual_tensor_1,text_tensor_1,category_tensor_1,visual_tensor_2,text_tensor_2,category_tensor_2,rating_2,label
        return visual_tensor_1,text_tensor_1,visual_tensor_2,text_tensor_2,rating_2,label

    def __len__(self):
        return self.len


def read_relationship(relation_type='bought_together'):
    with open(join(root, 'item_relation.pickle'), 'rb') as f:
        re_dic = pickle.load(f)
        return re_dic[relation_type]


def read_data_available():
    with open(join(root, 'available_asin_set.pickle'), 'rb')as f:
        return pickle.load(f)


def make_train_sample(relationship, asin_set, num=33000):
    '''
    从relationship中随机抽取关系作为训练集,剩余作为测试集
    :param relationship:
    :return: train_set,test_set
    '''
    asin_list = list(asin_set)
    shuffle(asin_list)
    relationship_list = []
    for item in relationship:
        for target in relationship[item]:
            relationship_list.append((item, target, 1))  # 原商品,关系品

    train_set = set(sample(relationship_list, num // 2))
    tmp_list = []
    p = 0
    for tup in train_set:
        item = tup[0]
        while True:
            target = asin_list[p]
            p += 1
            if p == len(asin_list):
                shuffle(asin_list)
                p = 0
            if target in relationship[item] or target not in asin_set:
                continue
            else:
                tmp_list.append((item, target, 0))
                break
    test_set = set(relationship_list) - train_set
    train_set = train_set | set(tmp_list)
    tmp_list = []
    for tup in test_set:
        item = tup[0]
        while True:
            target = asin_list[p]
            p += 1
            if p == len(asin_list):
                shuffle(asin_list)
                p = 0
            if (target in relationship[item] and (item, target, 0) not in train_set) or target not in asin_set:
                continue
            else:
                tmp_list.append((item, target, 0))
                break
    test_set = test_set | set(tmp_list)
    keys = set([tup[i] for tup in train_set | test_set for i in range(2)])

    return train_set, test_set, keys


def make_dataset_trick(asin_set, num=33000, save=True):
    with open(join(root, 'substitutes.pickle'), 'rb') as f:
        substitutes = pickle.load(f)
    with open(join(root, 'compliments.pickle'), 'rb')as f:
        compliments = pickle.load(f)
    substitutes_set = set(substitutes)
    compliments_set = set(compliments)
    all_list = []
    # 暴力测试
    # num=min(len(substitutes),len(compliments))
    substitutes_sample = sample(substitutes, num)
    compliments_sample = sample(compliments, num)
    all_list.extend([(tup[0], tup[1], 0) for tup in substitutes_sample])
    all_list.extend([(tup[0], tup[1], 1) for tup in compliments_sample])  # 搭配品推荐

    shuffle(all_list)
    train_list = all_list
    #valid_list = all_list[int(num*0.9):]
    test_comp = [(tup[0], tup[1], 1)
                 for tup in sample(list(compliments_set - set(compliments_sample)), int(num * 0.1))]
    test_subs = [(tup[0], tup[1], 0)
                 for tup in sample(list(substitutes_set - set(substitutes_sample)), int(num * 0.05))]
    asin_list = list(asin_set)
    sample_index = [randrange(0, len(asin_list)) for i in range(2 * int(num * 0.05))]

    test_no = [(asin_list[sample_index[i * 2]], asin_list[sample_index[i * 2 + 1]], 0) for i in range(int(num * 0.05))
               if (asin_list[sample_index[i * 2]], asin_list[sample_index[i * 2 + 1]]) not in substitutes_set and
               (asin_list[sample_index[i * 2]], asin_list[sample_index[i * 2 + 1]]) not in compliments_set]

    test_set = set(test_comp) | set(test_subs) | set(test_no)
    test_list=list(test_set)

    keys = []
    keys.extend([tup[i] for tup in all_list for i in range(2)])
    keys.extend([tup[i] for tup in test_list for i in range(2)])
    keys = set(keys)

    shuffle(test_list)
    valid_list=test_list[:len(test_list)//2]
    test_list=test_list[len(test_list)//2:]

    if save:
        with open('data_keys', 'wb')as f:
            pickle.dump((train_list,valid_list, test_list, keys), f)
    return train_list,valid_list, test_list, keys

def envolu_data_sample(asin_set, num=33000):
    with open(join(root, 'substitutes.pickle'), 'rb') as f:
        substitutes = pickle.load(f)
    with open(join(root, 'compliments.pickle'), 'rb')as f:
        compliments = pickle.load(f)
    dataMap={}
    substitutes_sample=sample(substitutes,num//2)
    compliments_sample=sample(compliments,num//2)
    for item in substitutes_sample:
        dataMap[(item[0],item[1])]=0
    for item in compliments_sample:
        dataMap[(item[0],item[1])]=1
    data_list=[]
    cnt=[0,0]
    keys=set()
    for key in dataMap:
        v=dataMap[key]
        keys.add(key[0])
        keys.add(key[1])
        data_list.append((key[0],key[1],v))
    shuffle(data_list)
    train_list=data_list[:int(num*0.8)]
    valid_list=data_list[int(num*0.8):int(num*0.9)]
    test_list=data_list[int(num*0.9):]
    return train_list,valid_list, test_list, keys


def read_rating():
    with open(join(root, 'item_rating.pickle'), 'rb') as f:
        return pickle.load(f)


def read_text_vec():
    with open(join(root, 'doc_vec_Electronics.pickle'), 'rb') as f:
        return pickle.load(f)


def read_category_vec():
    with open(join(root, 'item_category_vector.pickle'), 'rb') as f:
        return pickle.load(f)


def read_visual(key_set):
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            if asin.decode('utf-8') not in key_set:
                continue

            feature = np.asarray(a.tolist())
            feature = feature.reshape((1, feature.shape[0]))

            # feature=np.zeros((1,4096))
            yield {'asin': asin, 'feature': feature}

    img_path = join('E:/', 'Machine Learning Data Temp','Electronics', 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    # debug
    # it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
    imagefeature = {}
    try:
        for item in it:
            asin = item['asin'].decode('utf-8')
            if asin in key_set:
                imagefeature[asin] = item['feature']
    except EOFError as e:
        pass
    return imagefeature

