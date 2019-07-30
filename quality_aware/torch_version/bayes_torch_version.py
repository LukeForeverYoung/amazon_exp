'''
    pytorch version of products
'''



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
    parser.add_argument('--category', type=int, default=1,
                        help='category.')
    return parser.parse_args()


import argparse
import torch
from quality_aware.torch_version.Network import Net_bayes, train_step, test_step, Net_BN_category
from os.path import join
from random import sample, shuffle
import numpy as np
import torch.utils.data.dataloader as DataLoader
import math
from random import randint
from quality_aware.torch_version.Data import ExtDataset, collate_fn, collate_with_cat_fn, read_data_available, \
    read_category_vec, read_relationship, make_dataset_trick, read_rating, read_text_vec, read_visual, \
    make_train_sample, envolu_data_sample

root = join('F://', 'Amazon Dataset', 'Electronics')


if __name__ == '__main__':
    args = parase_args()

    # 调参
    args.batchsize=16
    args.learningrate=0.001

    asin_set = read_data_available()
    if args.category:
        category_feature = read_category_vec()
        asin_set=asin_set&category_feature.keys()
    else:
        category_feature=None
    relationship = read_relationship('bought_together')

    train_set, test_set, keys = make_train_sample(relationship, asin_set)
    test_list=list(test_set)
    train_list=list(train_set)
    valid_list=train_list[:int(len(train_list)*0.1)]
    train_list=train_list[int(len(train_list)*0.1):]

    #train_list,valid_list, test_list, keys = make_dataset_trick(asin_set)
    #train_list, valid_list, test_list, keys = envolu_data_sample(None)
    rating = read_rating()
    text_feature = read_text_vec()
    visual_feature = read_visual(keys)


    # 训练部分

    net = Net_BN_category() if (args.category) else Net_bayes()

    net.cuda()

    optimizer = torch.optim.RMSprop(params=net.parameters(), lr=args.learningrate)


    train_dataset = ExtDataset(train_list, text_feature, visual_feature, rating,category=category_feature)
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,pin_memory=True,
                                       collate_fn=(collate_with_cat_fn if (args.category) else collate_fn))

    valid_dataset = ExtDataset(valid_list,text_feature, visual_feature, rating,category_feature)
    valid_dataloader = DataLoader.DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False,
                                       collate_fn=(collate_with_cat_fn if (args.category) else collate_fn))


    test_dataset = ExtDataset(test_list, text_feature, visual_feature, rating,category_feature)
    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False,
                                             collate_fn=(collate_with_cat_fn if (args.category) else collate_fn))
    ep=0
    pre_acc=0
    useless=0
    best_ep=None
    while True:
        # 训练部分
        net.train()
        print('train step')
        loss = 0
        test_c=0
        for i, item in enumerate(train_dataloader):
            loss += train_step(item, optimizer, net, add_category=args.category)
            test_c+=1
            if test_c==825:
                print('mid loss:',loss)
                loss=0
                test_c=0

        # 验证部分
        net.eval()
        print('valid step')
        acc_list = []
        rand_acc_list=[]
        predict_positive=0
        positive=0
        for i, item in enumerate(valid_dataloader):
            acc_res,predict_pos,all_pos=test_step(item, net, add_category=args.category)
            predict_positive+=predict_pos
            positive+=all_pos
            acc_list.extend(acc_res)

            if args.category:
                label = item[7]
            else:
                label = item[5]
            print(label.shape)
            rand_acc_list.extend([randint(0,1) == label[j] for j in range(len(label))])
        acc_rate = np.mean(acc_list)
        print('ep:', ep, '\t','valid acc:{:.4f}%'.format(acc_rate),'\t',
              'recall rate:{:.2f}%'.format(predict_positive/positive*100.0),'\t',
              'random acc:{:.2f}%'.format(np.mean(rand_acc_list)))
        if acc_rate>pre_acc:
            best_ep=ep
            torch.save(net.state_dict(),join(root,'ep_{0}.torch'.format(ep)))
            useless=0
            pre_acc=acc_rate
        else:
            useless+=1
        if useless>=15:
            break
        ep += 1
    best_model=torch.load(join(root,'ep_{0}.torch'.format(best_ep)))
    net.load_state_dict(best_model)


    if args.category:
        torch.save(best_model, join(root, 'best_model_cat.torch'))
    else:
        torch.save(best_model,join(root,'best_model.torch'))
    net = Net_BN_category() if (args.category) else Net_bayes()
    net.eval()
    net.cuda()
    net.load_state_dict(best_model)

    # 计算准确率
    dataset = ExtDataset(list(test_set), text_feature, visual_feature, rating,category_feature)
    dataloader = DataLoader.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, collate_fn=(collate_with_cat_fn if(args.category) else collate_fn))
    print('test step')
    acc_list = []
    predict_positive = 0
    positive = 0
    for i, item in enumerate(dataloader):
        acc_res, predict_pos, all_pos = test_step(item, net, add_category=args.category)
        predict_positive += predict_pos
        positive += all_pos
        acc_list.extend(acc_res)
    acc_rate = np.mean(acc_list)
    print('test acc:', acc_rate,'\t','recall rate:{:.2f}%'.format(predict_positive/positive*100.0))
