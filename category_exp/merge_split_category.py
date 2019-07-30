import itertools
import pickle
import re
from collections import defaultdict
from os.path import join

from nltk.corpus import stopwords
from random import randint,sample

from numpy import mean

from category_exp.category_tree_defination import Node
from category_exp.nlp_util import category_fix, title_fix
from category_exp.word2vec_util import similarity, get_vector, get_group_vector
from public_utils import deep_flatten

root = join('F://', 'Amazon Dataset', 'Electronics')
stop_words = set(stopwords.words('english'))

#
#   对于category的处理
#   首先自底向上分割
#   需要先通过split('&')把高阶类别树拆成子部件,
#   对每个子部件再进行分词和进一步处理的操作,得到以词干构成的子部件.子部件不再拆分,
#   作为category的语义表示.将整棵树自底向上的以子部件为单位拆分成新树.
#
#
with open(join(root,'title.pickle'),'rb')as f:
    titles=pickle.load(f)

def split_tree(now_node):
    '''
    自底向上修正类别树
    对于每一个结点,基于&符号拆分成若干个sub结点(需修改key_text内容),并将原children分配到各个sub结点中,将sub结点返回到上层结点
    切割后的短语用_替代空格,在word2vec检索时再拆分出来
    :param now_node: 当前节点
    :return:
    '''
    child_list=[]
    for next_node in now_node.children:
        child_list.extend(split_tree(now_node.children[next_node]))
    text_node_split=[Node(word)for word in category_fix(now_node.text)]
    for sub_node in text_node_split:
        # 先使用list把map的子节点存下来,因为可能存在分配进来的子节点具有相同的key,使用自上而下的合并方法效率更高
        sub_node.children=defaultdict(list)
    # 如果目录没有足够的key,需要额外处理
    assert len(text_node_split)>0

    if len(text_node_split)==1:
        text_node_split[0].asin=now_node.asin
    else:
        # category结点带有商品,且被分割. 需要利用商品title和category的搭配性进行分配
        for item in now_node.asin:
            # 对于有类别无title的商品直接过滤
            if item not in titles:
                continue

            fixed_title=title_fix(titles[item])
            fixed_title=deep_flatten([get_vector(word, return_str=True)
                                      for word in fixed_title
                                      if get_vector(word, return_str=True) is not None])
            #print(titles[item],fixed_title)

            if len(fixed_title)==0:
                # 如果无论如何都求不到词向量,就随机分配
                assign=randint(0,len(text_node_split)-1)
                text_node_split[assign].asin.add(item)
                continue
            comp_list = [(similarity(sub_node.text.split('_'),fixed_title), i) for i, sub_node in
                         enumerate(text_node_split)]
            comp_list.sort(key=lambda tup: tup[0], reverse=True)
            parent_idx = comp_list[0][1]
            text_node_split[parent_idx].asin.add(item)

    for child in child_list:
        comp_list = [(similarity(sub_node.text.split('_'),child.text.split('_')),i)for i,sub_node in enumerate(text_node_split)]
        comp_list.sort(key=lambda tup:tup[0],reverse=True)
        parent_idx=comp_list[0][1]
        '''
        comp_list = [(similarity(text_node_split[parent_idx].text.split('_'),word),word)for i,word in enumerate(child.text.split('_'))]
        comp_list.sort(key=lambda tup: tup[0], reverse=True)
        max_sim=comp_list[0][0]
        comp_list=[tup[1] for tup in comp_list if tup[0]*2>=max_sim]
        child.tag=child.text
        child.text='_'.join(comp_list)
        '''
        text_node_split[parent_idx].children[child.text].append(child)

    return text_node_split


def dfs(now_node,tab,item_vec_array,path_vec,need_print=True):
    if need_print:
        print("\t"*tab,now_node.text)
    if tab!=0:
        path_vec.append(get_group_vector(now_node.text.split('_')))
        for item in now_node.asin:
            item_vec_array[item].extend(path_vec.copy())
    leaf_sum=len(now_node.asin)
    for next_node in now_node.children:
        for c in now_node.children[next_node]:
            leaf_sum+=dfs(c,tab+1,item_vec_array,path_vec.copy(),need_print=need_print)
    del path_vec
    return leaf_sum

if __name__=='__main__':

    with open(join(root, 'item_category_vector.pickle'), 'rb') as f:
        tmp=pickle.load(f)
        ss=set()
        for k in tmp:
            ss.add(tmp[k].shape)
        print(len(ss))
        input()

    with open('category_tree','rb') as f:
        root_node=pickle.load(f)
        root_node=split_tree(root_node)[0]
        # 存储商品的向量表示,因为商品会具有多个向量表示,所以先用list存储最后做均值
        item_vec_array=defaultdict(list)
        print(dfs(root_node,0,item_vec_array,[]))

        item_vec={}
        for item in item_vec_array:
            item_vec[item]=mean(item_vec_array[item],axis=0)
        with open(join(root,'item_category_vector.pickle'),'wb') as f:
            pickle.dump(item_vec,f)
