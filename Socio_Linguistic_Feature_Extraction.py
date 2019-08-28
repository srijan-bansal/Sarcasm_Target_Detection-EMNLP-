# This script extracts socio-linguistic features such as Part-of-Speech (POS), Name-Entity-Recognition (NER), Empath and LIWC
# Extracted features are dumbed into a pickle file 'Data_Aug.pkl'

import pickle
import pandas as pd
import numpy as np
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from Liwc_Trie_Functions import create_trie, get_liwc_categories
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize
from empath import Empath

df=pd.read_csv('dataset.csv')
ppd=pd.read_csv('pre_processed_dataset.csv')

ohe=OneHotEncoder()
lb=LabelEncoder()

# Using Stanford NER Tagger API
jar_n = '/localhome/debarshi/sarcasm/stanford-ner-2018-10-16/stanford-ner-3.9.2.jar'
model_n = '/localhome/debarshi/sarcasm/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
ner_tagger = StanfordNERTagger(model_n, jar_n, encoding='utf8')

# Using Stanford POS Tagger API
jar = '/localhome/debarshi/sarcasm/stanford-postagger-2018-10-16/stanford-postagger-3.9.2.jar'
model = '/localhome/debarshi/sarcasm/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')

# Extracting POS Features
POS_snippets=[]
for i in range(len(df['Snippet'])):
    POS_snippets.extend(pos_tagger.tag(word_tokenize(df['Snippet'][i])))
POS_snippets_type=[x[1] for x in POS_snippets]
POS_snippets_type=lb.fit_transform(POS_snippets_type)	
pos_vec=ohe.fit_transform(np.reshape(POS_snippets_type,(-1, 1)))
pos_vec=pos_vec.todense()

# Extracting NER Features
ner_snippets=[]
for i in range(len(df['Snippet'])):
    ner_snippets.extend(ner_tagger.tag(word_tokenize(df['Snippet'][i])))
ner_snippets_type=[x[1] for x in ner_snippets]
ner_snippets_type=lb.fit_transform(ner_snippets_type)	
ner_vec=ohe.fit_transform(np.reshape(ner_snippets_type,(-1, 1)))
ner_vec=ner_vec.todense()

# Extracting Empath Features
lexicon = Empath()
empath_vec=[]
for text in ppd['Candidate_words']:
    a=lexicon.analyze(text, normalize=True)
    bv=[]
    for i in a.values():
        bv.append(i)
    empath_vec.append(bv)

# Extracting LIWC Features
liwc_dict = pickle.load(open("LIWC_DIC",'rb'))
T = create_trie(liwc_dict)
liwx_list=[]
for i in range(len(ppd['Candidate_words'])):
     liwx_list.extend(get_liwc_categories(T,ppd['Candidate_words'][i]))
liwx_list=list(dict.fromkeys(liwx_list))
convertor={}
z=0
for i in liwx_list:
    convertor[i]=z
    z+=1
liwx_train=[]
for i in range(len(ppd['Candidate_words'])):
     liwx_train.append(list(get_liwc_categories(T,ppd['Candidate_words'][i])))
liwx_vec=np.zeros((len(ppd['Candidate_words']),64))
for (i,j) in enumerate(liwx_train):
    for k in j:
        liwx_vec[i][convertor[k]]=1

#Dumping extracted features in a pickle file 
f = open(b"Data_aug.pkl","wb")
pickle.dump(zip(pos_vec, ner_vec, liwx_vec,empath_vec),f,protocol = 2)
