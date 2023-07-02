import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt
from utils import*
import pickle
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Input, Layer,Dense, Activation,LSTM,GRU,Add,concatenate,Conv2D, MaxPooling2D,AveragePooling2D, Flatten,Dropout
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras_resnet

time_step = 100
w = 28    
num_PC = 3
israndom = True
randtime = 1


for r in range(0,randtime):    


    ssan_9_tr_pc_pkl = './vis_pkl/ssan_tr_pc.pkl'
    ssan_9_te_pc_pkl = './vis_pkl/ssan_te_pc.pkl'
    ssan_9_tr_pu_pkl = './vis_pkl/ssan_tr_pu.pkl'
    ssan_9_te_pu_pkl = './vis_pkl/ssan_te_pu.pkl'    

    dataID=1
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom)
    X_train,XP_train,Y_train,label = data[1],data[3],data[5]-1,data[6]       
    num_classes = Y_train.max()+1
    nb_features = X_train.shape[-1]    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, num_classes)
    y_train = np.squeeze(y_train)        
    ###############test#################
    dataID=2
    data_te = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom)
    X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
    n_classes = Y_test.max()+1
    y_test = np_utils.to_categorical(Y_test, n_classes)
    y_test = np.squeeze(y_test)

    
    model = load_model('./ssan_9_pcpu.h5')
    joint = Model(inputs=model.input,outputs=model.get_layer('JOINTDENSE').output)
    vis_feats = joint.predict([X_train,XP_train])    
    test_feats = joint.predict([X_test,XP_test])
    print(vis_feats.shape)
   
    y = y_train.argmax(axis=-1)
    y = y.tolist()
    outtt=[]
    for i,label_n in enumerate(y):
        outtt.append((label[label_n],vis_feats[i]))  
    print(len(outtt))

    y_te = y_test.argmax(axis=-1)
    y_te = y_te.tolist()
    testtt=[]
    for i,label_n in enumerate(y_te):
        testtt.append((label_test[label_n],test_feats[i]))   
    print(len(testtt))    

    pklname = ssan_9_tr_pu_pkl
    f = open(pklname,'wb')
    pickle.dump(outtt,f)
    f.close()
    
    pklname = ssan_9_te_pc_pkl
    f = open(pklname,'wb')
    pickle.dump(testtt,f)
    f.close()

    
