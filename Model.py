# This script implements various deep-neural LSTM architectures (Unidirectional, Bidirectional, Target-dependent LSTM) with different word embeddings for sarcasm target detection. 


import pandas as pd
import numpy as np
from keras.layers import Dense ,LSTM,concatenate,Input,Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pickle


df=pd.read_csv('dataset.csv')
ppd=pd.read_csv('pre_processed_dataset.csv')

f = open(b"Data_fast.pkl","rb")
keras_left_context, keras_right_context, keras_middle, labels = zip(*pickle.load(f))
f.close()

f = open(b"Data_aug.pkl","rb")
pos_vec, ner_vec, liwc_vec,empath_vec = zip(*pickle.load(f))

#print(np.shape(keras_left_context))
#print(np.shape(keras_right_context))
#print(np.shape(keras_middle))
#print(np.shape(pos_vec))
#print(np.shape(ner_vec))
#print(np.shape(liwc_vec))
#print(np.shape(empath_vec))

#Tuned-Hyper Parameters
embed_size = 1024
hidden_size = 32
num_epochs=30
layer_size = 16
batch_size = 64
mode = 'Uni' # 'Uni' : Unidirectional LSTM  |  'Bi' : Bidirectional LSTM  | 'TD' : Target-dependent LSTM
augmentation = False

# Function to compress candidate-words belonging to the same tweet/line togerather
def compress():
    lengths = []
    for i in range (1,len(ppd)):
        if ppd['left_context'][i][2:-2] == "<start>":
            lengths.append(i)
    lengths.append(len(ppd))
    compressor = []
    compressor.append(range(lengths[0]))
    for i in range (1,len(lengths)):
        compressor.append(range(lengths[i-1],lengths[i]))
    return compressor
comp = compress()

# Dataset division for 3-fold cross validation
indices = list (range (len(comp)))
np.random.shuffle(indices)
bins = []
bins.append(indices[:int(0.33*len(indices))])
bins.append(indices[int(0.33*len(indices)):int(0.66*len(indices))])
bins.append(indices[int(0.66*len(indices)):])

# Training Data, Test Data Preparation

def prep (train_indices, test_indices):
    print (len(train_indices), len(test_indices))

    train_ids = []
    for i in range(len(train_indices)):
        train_ids.extend(comp[train_indices[i]])

    test_ids = []
    for i in range(len(test_indices)):
        test_ids.extend(comp[test_indices[i]])

    # Training Data Preparion
    train_left   = []
    train_right  = []
    train_middle = []
    train_labels = []
    train_pos_vec= []
    train_neg_vec= []
    train_liwc_vec=[]
    train_empath_vec=[]

    for id in train_ids:
        train_left.append(keras_left_context[id])
        train_right.append(keras_right_context[id])
        train_middle.append(keras_middle[id])
        train_labels.append(labels[id])
        train_pos_vec.append(pos_vec[id])
        train_neg_vec.append(ner_vec[id])
        train_liwc_vec.append(liwc_vec[id])
        train_empath_vec.append(empath_vec[id])

    train_left   = np.array(train_left)
    train_right  = np.array(train_right)
    train_middle = np.array(train_middle)
    train_labels = np.array(train_labels)
    train_middle = np.expand_dims(train_middle,axis=1)
    train_pos_vec= np.array(train_pos_vec)
    train_neg_vec= np.array(train_neg_vec)
    train_liwc_vec= np.array(train_liwc_vec)
    train_empath_vec= np.array(train_empath_vec)

    # Training Data Preparion
    val_left   = []
    val_right  = []
    val_middle = []
    val_labels = []
    val_pos_vec= []
    val_neg_vec= []
    val_liwc_vec=[]
    val_empath_vec=[]

    for id in test_ids:
        val_left.append(keras_left_context[id])
        val_right.append(keras_right_context[id])
        val_middle.append(keras_middle[id])
        val_labels.append(labels[id])
        val_pos_vec.append(pos_vec[id])
        val_ner_vec.append(ner_vec[id])
        val_liwc_vec.append(liwc_vec[id])
        val_empath_vec.append(empath_vec[id])

    val_left   = np.array(val_left)
    val_right  = np.array(val_right)
    val_middle = np.array(val_middle)
    val_labels = np.array(val_labels)
    val_middle = np.expand_dims(val_middle, axis=1)
    val_pos_vec=np.array(val_pos_vec)
    val_ner_vec=np.array(val_ner_vec)
    val_liwc_vec=np.array(val_liwc_vec)
    val_empath_vec=np.array(val_empath_vec)

    # Below part only for TD lstm
    if mode == 'TD':	    
    	train_left = np.concatenate((train_left, train_middle), axis=1)
	train_right = np.concatenate((train_middle, train_right), axis=1)
    	val_left = np.concatenate((val_left, val_middle), axis=1)
    	val_right = np.concatenate((val_middle, val_right), axis=1)
    return(train_left,train_right,train_middle,train_pos_vec,train_ner_vec,train_liwc_vec,train_empath_vec,train_labels,val_left,val_right,val_middle,val_pos_vec,val_neg_vec,val_liwc_vec,val_empath_vec,val_labels)


def de_comp(arr, test_indices):
    arr = deque(arr)
    fin = []
    for i in test_indices:
        temp = []
        for j in range(len(comp[i])):
            temp.append(arr.popleft())
        fin.append(temp)
    return(fin)    

def accuracy (pred, labels, test_indices):
    pred = pred[0]
    num_sent = len(comp)
    num_words = len(pred)
    threshold = 0
    cnt = 0
    # Threshold calculation for binary classification problem
    for a,b in zip (pred,labels):
        if b==1.0:
            threshold+=a
            cnt+=1
    threshold = threshold.item()/cnt

    pred_th = []
    for x in pred:
        if (x<=threshold):
            pred_th.append(0)
        else :
            pred_th.append(1)

    pred_th = np.array(pred_th)
    print ("Number of Test sentences : {}".format(len(test_indices)))
    error = pred_th-labels
    error_d  = de_comp(error,test_indices)
    labels_d = de_comp(labels,test_indices)
    pred_d   = de_comp(pred_th,test_indices)
    em_cnt = 0
    ds_cnt = 0
    mic_f1 = 0

    for err in error_d:
        if (sum(err)==0):
           em_cnt += 1
        ds_cnt += float(len(err)-sum(np.abs(err)))/len(err)

    for lab, pre in zip(labels_d,pred_d):

        tp = 0
        fp = 0
        fn = 0
        tn = 0
              
        for i,j in zip(lab,pre):
            if (int(i) == 1)  and (int(j) ==0) :
                fn += 1
            elif (int(i) == 0)  and (int(j) ==1) :
                fp += 1
            elif (int(i) == 0)  and (int(j) ==0) :
                tn += 1
            elif (int(i) == 1) and (int(j) ==1):
                tp += 1
        try : 
            mic_f1 += float(2*tp) / (2*tp + fn +fp)
        except :
            pass

    TP=0
    TN=0
    FP=0
    FN=0
    for a,b in zip(labels, pred_th):

        if int(a)==0 and b==0:
            TN+=1
        if int(a)==1 and b==1:
            TP+=1
        if int(a)==0 and b==1:
            FP+=1
        if int(a)==1 and b==0:
            FN+=1 
    print ("TP = {}, TN = {},FP = {}, FN = {}".format(TP,TN,FP,FN))
    F1 = float(2*TP)/(2*TP + FP+ FN)
    EM = float(em_cnt)/len(test_indices)
    DS = float(ds_cnt)/len(test_indices)
    uF1= float(mic_f1)/len(test_indices)
    print ("EM Accuracy : {}".format(EM))
    print ("DS Accuracy : {}".format(DS))
    print ("Micro F1    : {}".format(uF1))
    print ("Macro F1 Score = {}".format(F1))
    return (pred_d, labels_d)

def model(train_l,train_r,train_m,train_pos,train_ner,train_liwc,train_empath,train_labels,test_l,test_r,test_m,test_pos,test_ner,test_liwc,test_empath,test_labels,test_indices):
    
    x=Input(shape=(None,embed_size))
    y=Input(shape=(None,embed_size))
    z=Input(shape=(None,embed_size))
    z1=Input(shape=([34]))
    z2=Input(shape=([4]))
    z3=Input(shape=([64]))
    z4=Input(shape=([194]))

    if mode == 'Bi':
    	left_out=Bidirectional(LSTM(hidden_size//2,return_sequences=False),input_shape=(train_l.shape[1:]))(x)      
    	middle = Bidirectional(LSTM(hidden_size//2,return_sequences=False),input_shape=(train_m.shape[1:]))(y)
    	right_out=Bidirectional(LSTM(hidden_size//2,return_sequences=False),input_shape=(train_r.shape[1:]))(z)

    else:
    	left_out  = LSTM(hidden_size,return_sequences=False)(x)
    	middle    = LSTM(hidden_size,return_sequences=False)(y)
    	right_out = LSTM(hidden_size,return_sequences=False)(y)

    pos_dense=Dense(32,activation='relu')(z1)
    ner_dense=Dense(16,activation='relu')(z2)
    liwc_dense=Dense(64,activation='relu')(z3)
    empath_dense=Dense(64,activation='relu')(z4)

    if mode == 'TD' and augmentation == False :
        out=concatenate([left_out,right_out],axis=-1)

    if mode == 'TD' and augmentation == True :
    	out=concatenate([left_out,right_out,pos_dense,ner_dense,liwc_dense,empath_dense],axis=-1)

    if mode != 'TD' and augmentation == False :
        out=concatenate([left_out,middle,right_out],axis=-1)

    if mode != 'TD' and augmentation == True :
    	out=concatenate([left_out,middle,right_out,pos_dense,ner_dense,liwc_dense,empath_dense],axis=-1)

    out=Dense(layer_size, activation='relu')(out)
    output=Dense(1, activation='sigmoid')(out)
    model = Model(inputs=[x,y,z,z1,z2,z3,z4], outputs=output)
    model.compile(optimizer=Adam(lr=10e-5),loss='binary_crossentropy',metrics=['accuracy'])
    print ("Starting Epochs")
    for i in range(num_epochs):
        model.fit([train_l,train_r,train_m,train_pos,train_ner,train_liwc,train_empath],train_labels,batch_size=batch_size, epochs=1,verbose=0)
        print('***************************************************************')
        print ("predicting_ Epoch : {}".format(i))
        pred_val=[]
        pred_val.append(model.predict([test_l,test_r,test_m,test_pos,test_ner,test_liwc,test_empath]))
        pre_d, lab_d = accuracy (pred_val, test_labels,test_indices)

        with open('Tweets Aug-{}.csv'.format(i), mode='w') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for a,b in zip (pre_d,lab_d):
                file_writer.writerow([a,b])
    return model
	
for i in range (3):
    print ("Fold {}".format(i+1))
    print (len(bins[0] + bins[1]),len(bins[2]))
    train_left,train_right,train_middle,pos_vec_train,ner_vec_train,liwc_vec_train,empath_vec_train,train_labels,val_left,val_right,val_middle,pos_vec_val,ner_vec_val,liwc_vec_val,empath_vec_val,val_labels = prep (bins[i%3] + bins[(i+1)%3], bins[(i+2)%3])
    Sar_model=model(train_left,train_right,train_middle,pos_vec_train,ner_vec_train,liwc_vec_train,empath_vec_train,train_labels,val_left,val_right,val_middle,pos_vec_val,ner_vec_val,liwc_vec_val,empath_vec_val,val_labels,bins[(i+2)%3])
    Sar_model.save_weights("Bert Tweets Aug.h5")
    print("Saved model to disk")
