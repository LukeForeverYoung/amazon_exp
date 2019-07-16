'''
    pytorch version of products
'''
import argparse
import pickle

import array
import torch
from sklearn.model_selection import KFold


def parase_args():
    parser = argparse.ArgumentParser(description="Run Encore.")
    parser.add_argument('--ImageDim', type=int, default=4096,
                        help='Image Dimension.')
    parser.add_argument('--TexDim', type=int, default=100,
                        help='Text Dimension.')
    parser.add_argument('--ImageEmDim', type=int, default=10,
                        help='Image Embedding Dimension.')
    parser.add_argument('--TexEmDim', type=int, default=10,
                        help='Text Embedding Dimension.')
    parser.add_argument('--HidDim', type=int, default=100,
                        help='Hidden Dimension.')
    parser.add_argument('--FinalDim', type=int, default=10,
                        help='Fainl Dimension.')
    parser.add_argument('--learningrate', type=float, default=0.001,
                        help='Learning Rate.')
    parser.add_argument('--trainchoice', nargs='?', default="Yes",
                        help='Training or Testing.')
    parser.add_argument('--epoch', type=int, default=30,
                        help='epoch.')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='batchsize.')
    return parser.parse_args()



from quality_aware.torch_version.Network import Net,train_step,test_step,predict
from os.path import join
from random import sample, shuffle
import numpy as np
from quality_aware.torch_version.Data import ExtDataset,collate_fn
import torch.utils.data.dataloader as DataLoader
import math
root = join('F://', 'Amazon Dataset', 'Electronics')


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
            if target in relationship[item]:
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
            if target in relationship[item] and (item, target, 0) not in train_set:
                continue
            else:
                tmp_list.append((item, target, 0))
                break
    test_set = test_set | set(tmp_list)
    keys = set([tup[i] for tup in train_set | test_set for i in range(2)])

    return train_set, test_set, keys


def read_rating():
    with open(join(root, 'item_rating.pickle'), 'rb') as f:
        return pickle.load(f)


def read_text_vec():
    with open(join(root, 'doc_vec_Electronics.pickle'), 'rb') as f:
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
            feature = feature.reshape((1,feature.shape[0]))

            #feature=np.zeros((1,4096))
            yield {'asin': asin, 'feature': feature}

    img_path = join('F://', 'FDMDownload', 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    # debug
    # it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
    imagefeature = {}
    try:
        for item in it:
            asin = item['asin'].decode('utf-8')
            if asin in asin_set:
                imagefeature[asin] = item['feature']
    except EOFError as e:
        pass
    return imagefeature


if __name__ == '__main__':
    args = parase_args()
    asin_set = read_data_available()
    relationship = read_relationship('bought_together')
    train_set, test_set, keys = make_train_sample(relationship, asin_set)

    rating = read_rating()
    text_feature = read_text_vec()
    visual_feature = read_visual(keys)
    train_list=list(train_set)
    test_list=list(test_set)
    del train_set,test_set
    net = Net(args.ImageDim, args.TexDim, args.ImageEmDim, args.TexEmDim, args.HidDim, args.FinalDim)
    net.cuda()
    optimizer = torch.optim.SGD(params=net.parameters(),lr=args.learningrate)
    kf = KFold(n_splits=5,shuffle=True)  # 把训练集分割成5份,交叉验证
    fold_state=[]

    for train_key,valid_key in kf.split(train_list):

        '''交叉验证需进行多轮,每轮划分出训练集和验证集'''
        # 训练部分
        #每一轮需要重置参数
        net.init_weight()
        sub_train=[train_list[k] for k in train_key.tolist()]
        dataset=ExtDataset(sub_train,text_feature,visual_feature,rating)
        dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=True,collate_fn=collate_fn)
        print('train step')
        for ep in range(args.epoch):
            loss=0
            for i,item in enumerate(dataloader):
                loss+=train_step(item,optimizer,net)
            print('ep:',ep,'\t','loss:',loss)

        # 验证部分
        sub_valid = [train_list[k] for k in valid_key.tolist()]
        dataset = ExtDataset(sub_valid, text_feature, visual_feature, rating)
        dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=False,collate_fn=collate_fn)
        print('valid step')
        acc_list=[]
        for i, item in enumerate(dataloader):
            acc_list.extend(test_step(item, net))
        acc_rate=np.mean(acc_list)
        print('valid acc:',acc_rate)
        fold_state.append((acc_rate,net.state_dict()))
    fold_state.sort(key=lambda item:item[0])
    best_model=fold_state[-1][1]
    torch.save(best_model,join(root,'best_model.torch'))
    net.load_state_dict(best_model)

    # 计算准确率
    dataset = ExtDataset(test_list, text_feature, visual_feature, rating)
    dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, collate_fn=collate_fn)
    print('test step')
    acc_list = []
    for i, item in enumerate(dataloader):
        acc_list.extend(test_step(item, net))
    acc_rate = np.mean(acc_list)
    print('test acc:', acc_rate)
