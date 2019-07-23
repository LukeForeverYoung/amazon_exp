import re

from nltk import tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
def category_fix(category):
    # category_fix是需要作为key的,所需最后需要用_连接到一起
    category=category.split('&')
    res_list=[]
    for sub_category in category:
        tokens = tokenize.word_tokenize(sub_category)
        new_cate = [word for word in tokens if word.replace('-','').isalnum() and not word in stop_words]
        if len(new_cate)>0:
            res_list.sort()
            res_list.append('_'.join(new_cate))
    return res_list

def title_fix(title):
    tokens = tokenize.word_tokenize(title)
    new_title = [word for word in tokens if word.replace('-','').isalnum() and not word in stop_words]
    for word in tokens:
        if len(re.split('[-/.]',word))>1:
            new_title.extend([s_word for s_word in re.split('[-/.]',word) if s_word.isalnum() and not s_word in stop_words])
    return list(set(new_title))