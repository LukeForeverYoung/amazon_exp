import pickle
from collections import defaultdict
from os.path import join
#import ujson as json
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import tokenize
from random import sample
from pattern3.text import singularize
from category_exp.category_tree_defination import Node,category_fix
import re
from category_exp.merge_split_category import similarity

#model=KeyedVectors.load(join('F:/','word2vec.840B.300d.bin'),mmap='r')
from category_exp.word2vec_util import word_in, get_vector

'''
def checker(str):
    if str in model.wv.vocab:
        print(str,'yes')
        return True
    else:
        print(str,'no')
    if str.lower() in model.wv.vocab:
        print(str.lower(),'yes')
        return True
    else:
        print(str.lower(), 'no')
    if singularize(str) in model.wv.vocab:
        print(singularize(str), 'yes')
        return True
    else:
        print(singularize(str), 'no')

    str=re.split('[^a-zA-Z0-9]',str)
    if len(str):
        res=[checker(s) for s in str]
        print(any(res))
        return any(res)
'''
import numpy as np
print('start:')
print(get_vector('apple'))
print(type(np.asarray(get_vector('apple'))))
while True:
    tmpa=input()
    print(get_vector('apple'))