import array
import pickle
from collections import defaultdict
from os.path import join

root = join('E://', 'Machine Learning Data Temp', 'Electronics')

def item_reader():
    filename='meta_Electronics.json'
    has_category=set()
    has_title=set()
    has_description=set()
    all_key=set()
    has_related=defaultdict(set)
    with open(join(root,filename),'r') as f:
        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=eval(tmp)
            asin=tmp['asin']
            all_key.add(asin)
            if 'categories' in tmp:
                has_category.add(asin)
            if 'title' in tmp:
                has_title.add(asin)
            if 'description' in tmp:
                has_description.add(asin)
            if 'related' in tmp:
                relate = tmp['related']
                if 'also_bought' in relate:
                    [has_related[asin].add(target) for target in relate['also_bought']]
                if 'bought_together' in relate:
                    [has_related[asin].add(target) for target in relate['bought_together']]
                if 'also_viewed' in relate:
                    [has_related[asin].add(target) for target in relate['also_viewed']]
                if 'buy_after_viewing' in relate:
                    [has_related[asin].add(target) for target in relate['buy_after_viewing']]
    return has_category,has_title,has_description,has_related,all_key

def review_reader():
    import ujson as json
    has_rating=defaultdict(int)
    has_review=defaultdict(int)
    has_summary=defaultdict(int)
    has_reviewer_id=defaultdict(int)
    i = 0
    with open(join(root, 'reviews_Electronics.json'), 'r')as f:
        while True:
            tmp = f.readline()
            if not tmp:
                break
            tmp = json.loads(f.readline())
            asin = tmp['asin']
            if 'reviewText' in tmp:
                has_review[asin]+=1
            if 'summary' in tmp:
                has_summary[asin]+=1
            if 'overall' in tmp:
                has_rating[asin]+=1
            if 'reviewerID' in tmp:
                has_reviewer_id[asin]+=1
    return has_rating,has_review,has_summary,has_reviewer_id

def visual_reader():
    '''
    图像特征无需预处理,此函数在使用visual_feature时帮助读入
    从预处理特征集中读取特征,将特征向量转化为numpy形式(节约内存)
    :param asin_set: 无残缺
    :return:
    '''
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield asin.decode('utf-8')

    img_path = join(root, 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    # debug
    # it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
    has_visual=set()
    try:
        for item in it:
            has_visual.add(item)
    except EOFError as e:
        pass
    return has_visual

def load_from_pickle():
    with open('all_key.pickle', 'rb')as f:
        all_key=pickle.load(f)
    with open('has_category.pickle', 'rb')as f:
        has_category=pickle.load(f)
    with open('has_title.pickle', 'rb')as f:
        has_title=pickle.load(f)
    with open('has_description.pickle', 'rb')as f:
        has_description=pickle.load(f)
    with open('has_related.pickle','rb')as f:
        has_related=pickle.load(f)
    with open('has_visual.pickle', 'rb')as f:
        has_visual=pickle.load(f)
    with open('has_rating.pickle', 'rb')as f:
        has_rating=pickle.load(f)
    with open('has_review.pickle', 'rb')as f:
        has_review=pickle.load(f)
    with open('has_summary.pickle', 'rb')as f:
        has_summary=pickle.load(f)
    with open('has_reviewer_id.pickle', 'rb')as f:
        has_reviewer_id=pickle.load(f)
    return all_key,has_category,has_title,has_description,has_related,\
            has_visual,has_rating,has_review,has_summary,has_reviewer_id

def filter_bad_relation(relate,available):
    print(len(available))
    print()
    res={}

    for key in relate:
        if key not in available:
            continue
        related_set=set()
        for target in relate[key]:
            if target in available:
                related_set.add(target)
        if len(related_set)!=0:
            res[key]=related_set

    return res

def print_line(text,value,down=498196):
    print(text,value,'//','{0:.4g}%'.format(100*value/down))

def relation_pair_sum(relation):
    sum=0
    for key in relation:
        sum+=len(relation[key])
    return sum
def display_hist(data,xlabel,ylabel,title,fontsize=20):
    # 设置matplotlib正常显示中文和负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


    related_num = np.asarray([len(data[k]) for k in data])
    hist=plt.hist(related_num, bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel(xlabel,fontsize=fontsize)
    # 显示纵轴标签
    plt.ylabel(ylabel,fontsize=fontsize)
    # 显示图标题
    plt.title(title,fontsize=fontsize)
    plt.show()
'''
has_category,has_title,has_description,has_related,all_key=item_reader()
print('all_key:',len(all_key))
print('has_category:',len(has_category))
print('has_title:',len(has_title))
print('has_description:',len(has_description))
has_visual=visual_reader()
print('has_visual:',len(has_visual))
has_rating,has_review,has_summary,has_reviewer_id=review_reader()
print('has_rating:',len(has_rating))
print('has_review:',len(has_review))
print('has_summary:',len(has_summary))
print('has_reviewer_id:',len(has_reviewer_id))
with open('all_key.pickle','wb')as f:
    pickle.dump(all_key,f)
with open('has_category.pickle','wb')as f:
    pickle.dump(has_category,f)
with open('has_title.pickle','wb')as f:
    pickle.dump(has_title,f)
with open('has_description.pickle','wb')as f:
    pickle.dump(has_description,f)
with open('has_visual.pickle','wb')as f:
    pickle.dump(has_visual,f)
with open('has_rating.pickle','wb')as f:
    pickle.dump(has_rating,f)
with open('has_review.pickle','wb')as f:
    pickle.dump(has_review,f)
with open('has_summary.pickle','wb')as f:
    pickle.dump(has_summary,f)
with open('has_reviewer_id.pickle','wb')as f:
    pickle.dump(has_reviewer_id,f)
'''
with open(join(root, 'available_asin_set.pickle'), 'rb')as f:
    data_available=pickle.load(f)
print(len(data_available))
all_key,has_category,has_title,has_description,has_related,\
            has_visual,has_rating,has_review,has_summary,has_reviewer_id=load_from_pickle()
has_text=has_description|has_title
filter_has_related=filter_bad_relation(has_related,has_visual&has_text&has_category)
print('所有商品:',len(all_key))
print('---meta属性部分----')
print_line('具有类别:',len(has_category))
print_line('具有标题:',len(has_title))
print_line('具有描述:',len(has_description))
print_line('存在文本表示(标题或描述)',len(has_text))
print_line('具有视觉表示:',len(has_visual))
print_line('具有文本&视觉表示:',len(has_visual&has_text))
print_line('具有文本&视觉&类别表示:',len(has_visual&has_text&has_category))
print_line('具有关系:',len(has_related))
print('关系对数:',relation_pair_sum(has_related))
print_line('具有关系(过滤掉缺失文本视觉类别)',len(filter_has_related))
print_line('关系对数(过滤后):',relation_pair_sum(filter_has_related),relation_pair_sum(has_related))

print('---review部分----')
print_line('具有评分:',len(has_rating))
print_line('具有评论:',len(has_review))
print_line('具有总结:',len(has_summary))
print_line('具有评论者ID:',len(has_reviewer_id))
print('-------')

from matplotlib import pyplot as plt
import numpy as np

# 随机生成（10000,）服从正态分布的数据
display_hist(filter_has_related,"关系数目","存在此数目的商品个数",'关系数(已过滤,忽略无关系商品)直方图')



