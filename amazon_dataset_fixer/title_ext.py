from os.path import join
import pickle
root = join('F://', 'Amazon Dataset', 'Electronics')


def item_reader():
    filename='meta_Electronics.json'
    category=[]

    titles={}
    with open(join(root,filename),'r') as f:
        while True:
            tmp=f.readline()
            if not tmp:
                break
            tmp=eval(tmp)
            if 'categories' in tmp and 'title' in tmp:
                titles[tmp['asin']]=tmp['title']
    return titles

titles=item_reader()
with open(join(root,'title.pickle'),'wb')as f:
    pickle.dump(titles,f)