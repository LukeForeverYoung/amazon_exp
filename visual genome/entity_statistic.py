import pickle
from collections import defaultdict
from os.path import join
import ujson as json
from gensim.models import KeyedVectors
from gensim.utils import SaveLoad
from gensim.scripts.glove2word2vec import glove2word2vec
root = join('F://', 'Visual Genome')
'''
entity_dict=defaultdict(int)
with open(join(root,'relationships.json'))as f:
    obj_list=json.load(f)
    for item in obj_list:
        relation_list=item['relationships']
        for relation in relation_list:
            entity_dict[relation['subject']['name']]+=1
            entity_dict[relation['object']['name']]+=1
with open('entity_dict.pickle','wb') as f:
    pickle.dump(entity_dict,f)
print(len(entity_dict.keys()))
'''
model=KeyedVectors.load_word2vec_format(join('F:/','word2vec.840B.300d.txt'))
model.save(join('F:/','word2vec.840B.300d.bin'))

print('ok')
input()
synset_dict={}
with open(join(root,'synsets.json'))as f:
    obj_list=json.load(f)
    for item in obj_list:
        synset_dict[item['synset_name'].split('.')[0]]=item['synset_definition']
print('len synset:',len(synset_dict.keys()))

not_contain=set()
for key in synset_dict:
    if key not in word_vec:
        print(key)
        not_contain.add(key)

print(len(not_contain))



