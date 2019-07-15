import datetime
import multiprocessing
import os
import time
from os.path import join
import numpy as np
from collections import defaultdict
import nltk
import ujson as json
from nltk import word_tokenize, porter, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import html
import pickle
def read_meta():
    root = join('F://', 'Amazon Dataset', 'Electronics')
    filename = 'fixed_meta_Electronics.json'
    with open(join(root, filename), 'r') as f:
        data=json.load(f)
    return data

def sort_with_asin(data):
    n_d=defaultdict(list)
    for item in data:
        n_d[item['asin']].append(item)
    return n_d

def preprocess(title,description):
    stop_words = set(stopwords.words('english'))
    stemmer=PorterStemmer()
    #在词的过滤部分,html标记语法会被过滤掉
    #title=html.unescape(html.unescape(title))
    #description=html.unescape(html.unescape(description))
    sents=[title]
    sents.extend(sent_tokenize(description))
    n_sents=[]
    for sent in sents:
        #print(sent)
        tokens = word_tokenize(sent, language='english')
        words = [stemmer.stem(word) for word in tokens if word.isalpha() and not word in stop_words]
        n_sents.extend(words)
    return n_sents


if __name__=='__main__':
    print(nltk.__version__)
    model = Doc2Vec(window=20, vector_size=100, workers=6)
    data=read_meta()
    data_size=len(data)
    print(data_size)
    documents=[]
    for i,prod in enumerate(data):
        #print('title:',prod['title'])
        #print('desc:',prod['description'])
        words=preprocess(prod['title'],prod['description'])
        documents.append(TaggedDocument(words, [i]))
        print(i,'/',data_size)
    model.build_vocab(documents)
    model.train(documents,start_alpha=0.1,total_examples=len(documents),epochs=10)
    fin_res={}
    for i,doc in enumerate(documents):
        prod_asin=data[i]['asin']
        res=model.infer_vector(doc.words)
        fin_res[prod_asin]=res

    root = join('F://', 'Amazon Dataset', 'Electronics')
    filename = 'doc_vec_Electronics.pickle'

    import pickle
    with open(join(root,filename), 'wb') as f:
        pickle.dump(fin_res,f)
    print('ok')



