import re

from nltk.stem import PorterStemmer
from nltk import tokenize
from random import sample
#import ujson as json
import numpy as np
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from os.path import join
from numpy import mean
from pattern3.text import singularize
from nltk.stem.wordnet import WordNetLemmatizer
from public_utils import deep_flatten
lmtzr = WordNetLemmatizer()
model=KeyedVectors.load(join('F:/','word2vec.840B.300d.bin'),mmap='r')

def get_group_vector(arr):
    res=[get_vector(a) for a in arr]
    return mean(res,axis=0)

def get_vector(s,return_str=False):
    '''
    使用多种方法尝试修正词汇并获取词向量,越靠前的方法越靠谱,因此尝试成功则会截断后续操作
    '''
    # 先尝试原字符串,和大小写\词态修正
    if s in model.wv.vocab:
        if return_str:
            return s
        return np.asarray(model.get_vector(s))
    if s.lower() in model.wv.vocab:
        if return_str:
            return s.lower()
        return np.asarray(model.get_vector(s.lower()))
    if lmtzr.lemmatize(s) in model.wv.vocab:
        if return_str:
            return lmtzr.lemmatize(s)
        return np.asarray(model.get_vector(lmtzr.lemmatize(s)))
    r_s=s
    # 如果不行可能是存在连字符或者其他分隔符号且保留的话无法索引到向量, split后用得到的序列求vec均值(方法借鉴gensim的多词相似度求解实现)
    # 直接用非字母/数字符号分离
    tmp = re.split('[^a-zA-Z0-9]', r_s)
    #print(s)
    if len(tmp)>1:
        res = [get_vector(sub_s,return_str) for sub_s in tmp if get_vector(sub_s,return_str) is not None]
        if len(res):
            if return_str:
                return res
            return mean(res,axis=0)
    # 以数字作为分离并保留数字作为词前缀,用以解决类似connect3d的词语
    tmp=re.split('[0-9]', r_s)
    if len(tmp)>1:
        res=[]
        pos=0
        for item in tmp:
            if pos!=0:
                item=r_s[pos]+item
            if get_vector(item, return_str) is not None:
                res.append(get_vector(item, return_str))
            pos+=len(item)
        if len(res):
            if return_str:
                return res
            return mean(res,axis=0)
    # 以驼峰式命名法进行分离
    tmp = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', r_s)
    tmp=[m.group(0) for m in tmp]
    if len(tmp)>1:
        res=[]
        for item in tmp:
            if get_vector(item, return_str) is not None:
                res.append(get_vector(item, return_str))
        if return_str:
            return res
        return mean(res,axis=0)
    return None

def word_in(single_word):
    return single_word in model.wv.vocab

def similarity(text_group_1,text_group_2):
    if type(text_group_1) is str:
        text_group_1=[text_group_1]
    if type(text_group_2) is str:
        text_group_2=[text_group_2]
    #print(text_group_1,text_group_2)
    text_group_1=deep_flatten([get_vector(str,return_str=True)for str in text_group_1])
    text_group_2=deep_flatten([get_vector(str,return_str=True)for str in text_group_2])
    #print(text_group_1,text_group_2)
    return model.n_similarity(text_group_1,text_group_2)