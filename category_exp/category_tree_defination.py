

from category_exp.nlp_util import category_fix


class Node():
    def __init__(self,text):
        self.number=0
        self.text=text
        #self.asin_set=set()
        self.children={}

        self.tag=""
        self.asin=set()
    def add(self,asin_id):
        self.asin.add(asin_id)
        self.number+=1
    def next(self,text):
        if text in self.children:
            return self.children[text]
        else:
            return None
    def add_child(self,child_node):
        self.children[child_node.text]=child_node





def dfs(now_node,tab,key_word_set=None,need_print=True):
    if need_print:
        print("\t"*tab,now_node.text,now_node.number)
    if not key_word_set is None and tab:
        for item in category_fix(now_node.text):
            for word in item:
                key_word_set.add(word)
    if len(now_node.children)==0:
        return 1
    leaf_sum=0
    for next_node in now_node.children:
        leaf_sum+=dfs(now_node.children[next_node],tab+1,key_word_set=key_word_set,need_print=need_print)
    return leaf_sum