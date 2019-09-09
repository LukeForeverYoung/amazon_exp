import pickle
from os.path import join
root = join('E://', 'Machine Learning Data Temp', 'Electronics')



with open(join(root, 'item_category.pickle'), 'wb') as f:
    categories=pickle.load(f)