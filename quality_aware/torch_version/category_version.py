'''
    pytorch version of products
'''
import argparse
import pickle

import array
import torch
from sklearn.model_selection import KFold
from copy import deepcopy
from random import randrange


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


from quality_aware.torch_version.Network import train_step, test_step, Net_category
from os.path import join
from random import sample, shuffle
import numpy as np
from quality_aware.torch_version.Data import ExtDataset, collate_fn, collate_with_cat_fn, read_data_available, \
    read_category_vec, read_relationship, make_dataset_trick, read_rating, read_text_vec, read_visual
import torch.utils.data.dataloader as DataLoader
import math

root = join('F://', 'Amazon Dataset', 'Electronics')





if __name__ == '__main__':
    args = parase_args()


    # 调参
    #args.learningrate=0.0005
    #args.epoch=50


    asin_set = read_data_available()
    category_feature = read_category_vec()
    asin_set = asin_set & category_feature.keys()
    relationship = read_relationship('bought_together')
    #train_set, test_set, keys = make_train_sample(relationship, asin_set)
    train_list, valid_list,test_list, keys = make_dataset_trick(asin_set)

    rating = read_rating()
    text_feature = read_text_vec()
    visual_feature = read_visual(keys)
    net = Net_category()
    net.cuda()
    # optimizer = torch.optim.SGD(params=net.parameters(),lr=args.learningrate)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.learningrate)
    kf = KFold(n_splits=5, shuffle=True)  # 把训练集分割成5份,交叉验证
    fold_state = []

    for train_key, valid_key in kf.split(train_list):

        '''交叉验证需进行多轮,每轮划分出训练集和验证集'''
        # 训练部分
        # 每一轮需要重置参数
        net.init_weight()
        sub_train = [train_list[k] for k in train_key.tolist()]
        dataset = ExtDataset(sub_train, text_feature, visual_feature, rating, category_feature)
        dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=True,
                                           collate_fn=collate_with_cat_fn)
        print('train step')
        for ep in range(args.epoch):
            loss = 0
            for i, item in enumerate(dataloader):
                loss += train_step(item, optimizer, net, add_category=True)
            print('ep:', ep, '\t', 'loss:', loss)

        # 验证部分
        sub_valid = [train_list[k] for k in valid_key.tolist()]
        dataset = ExtDataset(sub_valid, text_feature, visual_feature, rating, category_feature)
        dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                                           collate_fn=collate_with_cat_fn)
        print('valid step')
        acc_list = []
        for i, item in enumerate(dataloader):
            acc_list.extend(test_step(item, net, add_category=True))
        acc_rate = np.mean(acc_list)
        print('valid acc:', acc_rate)
        fold_state.append((acc_rate, deepcopy(net.state_dict())))
        break
    fold_state.sort(key=lambda item: item[0])
    best_model = fold_state[-1][1]
    torch.save(best_model, join(root, 'best_model.torch'))
    net.load_state_dict(best_model)

    # 计算准确率
    dataset = ExtDataset(valid_list, text_feature, visual_feature, rating, category_feature)
    #dataset = ExtDataset(test_list, text_feature, visual_feature, rating, category_feature)
    dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                                       collate_fn=collate_with_cat_fn)
    print('test step')
    acc_list = []
    for i, item in enumerate(dataloader):
        acc_list.extend(test_step(item, net, add_category=True))
    acc_rate = np.mean(acc_list)
    print('test acc:', acc_rate)
