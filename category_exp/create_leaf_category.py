
import pickle
from os.path import join
#import ujson as json
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import tokenize
from random import sample
from category_exp.category_tree_defination import Node,dfs
from pattern3.text import singularize
root = join('F://', 'Amazon Dataset', 'Electronics')


'''


with open('category.pickle','rb') as f:
    
    stemmer = PorterStemmer()
    category=pickle.load(f)
    for cate in category:
        tokens=tokenize.word_tokenize(cate)
        #new_cate=[stemmer.stem(word) for word in tokens if word.isalpha() and not word in stop_words]
        #使用词干提取得到的是词干而不是词,不便于索引
        new_cate = [word for word in tokens if word.isalpha() and not word in stop_words]
        cate_list.append((new_cate,cate))
        #print(new_cate,cate)

for key in cate_list:
    print(key)
    input()
    '''




root_node=Node("root node")

def insert(cat_seq,asin):
    now_node=root_node
    for cat_name in cat_seq:
        if not now_node.next(cat_name):
            now_node.add_child(Node(cat_name))
        now_node = now_node.next(cat_name)
    now_node.add(asin)

def reader():
    root=join('F://','Amazon Dataset','Electronics')
    filename='meta_Electronics.json'
    category=[]
    with open(join(root,filename),'r') as f:

        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=eval(tmp)
            if 'categories' in tmp:
                for cat_seq in tmp['categories']:
                    # make category tree
                    insert(cat_seq,tmp['asin'])
                    for cat in cat_seq:
                        # make category list
                        category.append(cat)
            '''
            if 'categories' in tmp and len(tmp['categories'])>=2 and 'title' in tmp:
                print(tmp)
                print(tmp['title'])
                input()
            '''
    return category

'''
key_word=set()
with open('category_tree','rb') as f:
    dfs(pickle.load(f),0,key_word_set=key_word,need_print=False)
print(len(key_word))
'''

category=reader()
print('read finish')
with open('category_tree','wb') as f:
    pickle.dump(root_node,f)
print('write finish')
'''
with open('category.txt','w') as f:
    f.writelines([cat+"\n" for cat in category])

with open('category.pickle','wb') as f:
    pickle.dump(category,f)

'''
#print('leaf sum',dfs(root_node,0))
#print('item_sum',len(category))
