# This script pre-processes the sarcasm target detection dataset for neural netowrk architecture
# Sarcasm Target Detection Dataset:https://github.com/Pranav-Goel/Sarcasm-Target-Detection named as 'dataset.csv' having text and target-status
# (Left-context,candidate word, right context,label) tuples are stored as 'pre_processed_dataset.csv'
# The code uses fasttext the embeddings (https://fasttext.cc/docs/en/english-vectors.html)
# One may use the same code for embeddings by Matthew E. Peters et al. (https://allennlp.org/elmo) or embeddings by Google (https://pypi.org/project/bert-embedding/)


import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
import codecs

tokenizer = RegexpTokenizer(r'\w+')
detokenizer = TreebankWordDetokenizer()
df=pd.read_csv('dataset.csv')
embedding_file = './crawl-300d-2M.vec'
 

def loadEmbed():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(embedding_file, encoding='utf-8')
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index

#Loading pre-trained embeddings
model=loadEmbed()
all_words=[]
for i in range(len(df['Snippet'])):
    all_words.extend(df['Snippet'][i].split())
all_words=list(dict.fromkeys(all_words))
all_words=[x.lower() for x in all_words]

embeddings={}
for each in all_words:
    if each.lower() not in model.keys():
        embeddings[each]=model['unk']
    else:
        embeddings[each]=model[each.lower()]

ppd=pd.read_csv('pre_processed_dataset.csv')
length_l=[]
for i in range(len(ppd['left_context'])):
    length_l.append(len(tokenizer.tokenize(ppd['left_context'][i])))
length_r=[]
for i in range(len(ppd['right_context'])):
    length_r.append(len(tokenizer.tokenize(ppd['right_context'][i])))

embeddings['<pad>']= [0]*300

# Embedding Left Context 
keras_left_context=[]
for i in range(len(ppd['left_context'])):
    one_vector=[]
    temp=tokenizer.tokenize(ppd['left_context'][i])
    one_vector.append(model['start'])
    for m in temp[1:]:
        a = embeddings[m.lower()]
        one_vector.append(a)
    one_vector.extend([embeddings['<pad>'] for x in range(78-length_l[i])])
    keras_left_context.append(one_vector)

# Embedding Right Context 
keras_right_context=[]
for i in range(len(ppd['right_context'])):
    one_vector=[]
    temp=tokenizer.tokenize(ppd['right_context'][i])
    for m in temp[1:]:
        one_vector.append(embeddings[m.lower()])
    one_vector.append(model['end'])
    one_vector.extend([embeddings['<pad>'] for x in range(78-length_r[i])])
    keras_right_context.append(one_vector)
keras_middle=[]

# Embedding Candidate Word
for i in range(len(ppd['Candidate_words'])):
    keras_middle.append(embeddings[ppd['Candidate_words'][i].lower()])

#Saving the processed dataset in a pickle file
f = open(b"Data_fast.pkl","wb")
pickle.dump(zip(keras_left_context,keras_right_context,keras_middle,ppd['target_status']),f)

