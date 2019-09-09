'''
    数据介绍:
        meta_data中包含的关键数据
            asin:商品id
            imUrl:图像链接
            title:标题
            description:商品描述
            related:商品关系(also bought,also viewed,bought together,buy after viewing)

        reviews中包含的关键数据
            reviewerID:评论者id
            asin:商品id
            reviewText:评论内容
            summary:总结
            overall:评分
        image_feature:来自Amazon Dataset的预处理特征
            asin:商品id
            feature_list:特征向量(数组形式)
    数据缺陷:
        1.商品的文字描述缺失
        2.商品的视觉特征缺失
        3.商品评分缺失
        4.商品的关系缺失(部分商品的关联商品可能属于残缺)
        ...待补充
    处理方法:
        1.首先统计text/visual/rating均完整的asin集合
        2.在抓取关系时,过滤掉数据不完整的商品,并仅保留具有关系的商品关系集
    注意:
        related为空的数据完整商品可能会被动存在于其他商品的关系集合中,因此其特征需要被保留.
        当在训练模型时,训练集的选择应该从筛选后的关系集中采样,而非从完整数据商品集中,否则训练集中会存在一定数量的空关系商品.
'''
import pickle
from os.path import join

import array
# 根目录,根据情况修改
root = join('F://', 'Amazon Dataset', 'Electronics')

def read_visual():
    '''
    读取具有视觉特征的asin集合
    :return: asin集合
    '''
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10)
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield {'asin': asin}
    img_path = join(root, 'image_features_Electronics.b')
    it = readImageFeatures(img_path)
    visual_asin=[]
    try:
        for item in it:
            asin = item['asin'].decode('utf-8')
            visual_asin.append(asin)
    except EOFError as e:
        pass
    return set(visual_asin)
def read_text():
    '''
    读取具有文本特征的asin集合
    此方法调用前需要抽取文本特征,并以{'asin':vector}的形式序列化至doc_vec_Electronics.pickle中
    '''
    with open(join(root, 'doc_vec_Electronics.pickle'), 'rb') as f:
        return pickle.load(f).keys()

def read_rating():
    '''
    读取具有rating的asin集合
    此方法调用前需要计算商品rating,并以{'asin':scalar}的形式序列化至doc_vec_Electronics.pickle中
    '''
    with open(join(root, 'item_rating.pickle'), 'rb') as f:
        return pickle.load(f).keys()

def read_category():
    with open(join(root, 'item_category.pickle'), 'rb') as f:
        return pickle.load(f).keys()

if __name__=='__main__':
    '''
        需要先使用某种方法获得visual/text feature和rating
    '''
    visual_available = read_visual()
    text_available = read_text()
    rating_available = read_rating()
    category_available = read_category()
    print(len(visual_available))
    print(len(text_available))
    print(len(rating_available))
    print(len(category_available))
    # len 277271
    data_available = visual_available & text_available & rating_available & category_available
    print(len(data_available))
    print('B00KPYMOL4' in data_available)
    print('B00KPYMOL4' in text_available)
    with open(join(root, 'available_asin_set.pickle'), 'wb')as f:
        pickle.dump(data_available, f)
    '''
        随后可以利用meta-data和available_asin_set抽取关系
    '''