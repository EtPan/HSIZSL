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

def RNN(time_step,nb_features,num_classes):
    RNNInput = Input(shape=(time_step,nb_features),name='RNNInput')
    RNNSpectral = GRU(256,name='RNNSpectral',consume_less='gpu',W_regularizer=l2(0.0001),U_regularizer=l2(0.0001))(RNNInput)  
    #RNNSpectral = Dropout(0.5)(RNNSpectral) 
    RNNDense = Dense(128,activation='relu', name='RNNDense')(RNNSpectral)   
    RNNDense = Dropout(0.4)(RNNDense)
    RNNSOFTMAX = Dense(num_classes,activation='softmax', name='RNNSOFTMAX')(RNNDense) 
    model = Model(inputs=[RNNInput],outputs=[RNNSOFTMAX])
    return model
    
def RESNET(num_PC,img_rows,img_cols,num_classes):
    model = keras_resnet.ResNet18((num_PC,img_rows,img_cols),classes=num_classes)
    #rmsp = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-05)    
    #adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999,decay=1e-2,epsilon=1e-05)
    #model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])    
    return model    

def CNN(num_PC,img_rows,img_cols,num_classes):
    CNNInput = Input(shape=[num_PC,img_rows,img_cols],name='CNNInput')
    CONV1 = Conv2D(32, (5, 5), activation="relu", name="CONV1", padding="same")(CNNInput) 
    POOL1 = MaxPooling2D((2, 2), strides=(2,2), name='POOL1')(CONV1)   
    CONV2 = Conv2D(64, (5, 5), activation="relu", name="CONV2", padding="same")(POOL1)
    POOL2 = MaxPooling2D((2, 2), strides=(2,2), name='POOL2')(CONV2)
    #CONV3 = Conv2D(128, (7, 7), activation="relu", name="CONV3", padding="same")(POOL2)
    #POOL3 = MaxPooling2D((2,2), strides=(2,2), name='POOL3')(CONV3)    
    FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    CNNDense = Dense(512,activation='relu', name='CNNDENSE0')(FLATTEN2)
    CNNDense = Dropout(0.6)(CNNDense)       
    CNNSOFTMAX = Dense(num_classes,activation='softmax', name='CNNSOFTMAX')(CNNDense)   
    model = Model(inputs=[CNNInput],outputs=[CNNSOFTMAX])
    return model
    
def SSUN(time_step,nb_features,num_PC,img_rows,img_cols,num_classes):    
    RNNInput = Input(shape=(time_step,nb_features),name='RNNInput')
    RNNSpectral = GRU(256,name='RNNSpectral',consume_less='gpu',W_regularizer=l2(0.0001),U_regularizer=l2(0.0001))(RNNInput)  
    #RNNSpectral = Dropout(0.2)(RNNSpectral) 
    RNNDense = Dense(128,activation='relu', name='RNNDense')(RNNSpectral)   
    #RNNSOFTMAX = Dense(nb_classes,activation='softmax', name='RNNSOFTMAX')(RNNDense)    
 
    CNNInput = Input(shape=[num_PC,img_rows,img_cols],name='CNNInput')
    CONV1 = Conv2D(32, (3, 3), activation="relu", name="CONV1", padding="same")(CNNInput) 
    POOL1 = MaxPooling2D((2, 2), strides=(2,2), name='POOL1')(CONV1)   
    CONV2 = Conv2D(64, (3, 3), activation="relu", name="CONV2", padding="same")(POOL1)
    POOL2 = MaxPooling2D((2, 2), strides=(2,2), name='POOL2')(CONV2)
    CONV3 = Conv2D(128, (3, 3), activation="relu", name="CONV3", padding="same")(POOL2)
    POOL3 = MaxPooling2D((2,2), strides=(2,2), name='POOL3')(CONV3)    
    FLATTEN1 = Flatten(name='FLATTEN1')(POOL1)
    #FLATTEN1 = Dropout(0.4)(FLATTEN1)
    FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    #FLATTEN2 = Dropout(0.4)(FLATTEN2) 
    FLATTEN3 = Flatten(name='FLATTEN3')(POOL3)    
    #FLATTEN3 = Dropout(0.4)(FLATTEN3) 
    DENSE1 = Dense(128,activation='relu', name='DENSE1')(FLATTEN1)
    DENSE2 = Dense(128,activation='relu', name='DENSE2')(FLATTEN2)
    DENSE3 = Dense(128,activation='relu', name='DENSE3')(FLATTEN3)        
    CNNDense = Add(name='CNNDense')([DENSE1, DENSE2, DENSE3])    
    #CNNSOFTMAX = Dense(nb_classes,activation='softmax', name='CNNSOFTMAX')(CNNDense)    
    JOINT = concatenate([RNNDense,CNNDense],axis=1)
    JOINT1 = Dropout(0.4)(JOINT)
    JOINTDENSE = Dense(128,activation='relu', name='JOINTDENSE')(JOINT1)
    #JOINTDENSE = Dropout(0.4)(JOINTDENSE)
    JOINTSOFTMAX = Dense(num_classes,activation='softmax',name='JOINTSOFTMAX')(JOINTDENSE)
    model = Model(inputs=[RNNInput,CNNInput], outputs=[JOINTSOFTMAX])#,RNNSOFTMAX,CNNSOFTMAX])  
    return model

def SSAN(time_step,nb_features,num_PC,img_rows,img_cols,num_classes):    
    RNNInput =Input(shape=(time_step,nb_features),name='RNNInput')
    RNNSpectral = GRU(256,name='RNNSpectral',consume_less='gpu',W_regularizer=l2(0.0001),U_regularizer=l2(0.0001))(RNNInput)  
    #RNNSpectral = Dropout(0.4)(RNNSpectral) 
    RNNDense1 = Dense(128,activation='relu', name='RNNDense')(RNNSpectral)
    #RNNDense1 = Dropout(0.6)(RNNDense1)#0.6
    #RNNDense = Dense(128,activation='relu', name='RNNDense')(RNNSpectral)   
    #RNNSOFTMAX = Dense(nb_classes,activation='softmax', name='RNNSOFTMAX')(RNNDense)    
 
    CNNInput = Input(shape=[num_PC,img_rows,img_cols],name='CNNInput')
    CONV1 = Conv2D(32, (5, 5), activation="relu", name="CONV1", padding="same")(CNNInput) 
    POOL1 = MaxPooling2D((2, 2), strides=(2,2), name='POOL1')(CONV1)   
    CONV2 = Conv2D(64, (5, 5), activation="relu", name="CONV2", padding="same")(POOL1)
    POOL2 = MaxPooling2D((2, 2), strides=(2,2), name='POOL2')(CONV2)
    #CONV3 = Conv2D(128, (7, 7), activation="relu", name="CONV3", padding="same")(POOL2)
    #POOL3 = MaxPooling2D((2,2), strides=(2,2), name='POOL3')(CONV3)    
    FLATTEN2 = Flatten(name='FLATTEN2')(POOL2)
    CNNDense0 = Dense(128,activation='relu', name='CNNDENSE0')(FLATTEN2)
    CNNDense0 = Dropout(0.4)(CNNDense0)
    #CNNDense1 = Dense(num_classes,activation='relu', name='CNNDENSE1')(CNNDense0)         
    #CNNSOFTMAX = Dense(nb_classes,activation='softmax', name='CNNSOFTMAX')(CNNDense)    
    
    #RNNDense2 = Dense(256,activation='relu', name='RNNDense2')(RNNDense1)
    #CNNDense2 = Dense(256,activation='relu', name='CNNDense2')(CNNDense1)
    JOINT = concatenate([RNNDense1,CNNDense0],axis=1)
    #JOINT = Dropout(0.4)(JOINT)
    JOINTDENSE = Dense(1024,activation='relu', name='JOINTDENSE')(JOINT)
    JOINTDENSE = Dropout(0.4)(JOINTDENSE)
    JOINTSOFTMAX = Dense(num_classes,activation='softmax',name='JOINTSOFTMAX')(JOINTDENSE)
    model = Model(inputs=[RNNInput,CNNInput], outputs=[JOINTSOFTMAX])#,RNNSOFTMAX,CNNSOFTMAX])  
    return model
    
time_step = 100
w = 28    
num_PC = 3
israndom = True
randtime = 5
nb_epoch = 30
batch_size = 256
OAJoint_Pavia = np.zeros((9+2,randtime))
    
for r in range(0,randtime):
    #################train#################
    #3&4 time_step = 4 
    dataID=2
    data = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom)
    X_train,XP_train,Y_train,label = data[1],data[3],data[5]-1,data[6]       
    num_classes = Y_train.max()+1
    nb_features = X_train.shape[-1]    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(Y_train, num_classes)
    y_train = np.squeeze(y_train)        
    ###############test#################
    dataID=1
    data_te = HyperspectralSamples(dataID=dataID, timestep=time_step, w=w, num_PC=num_PC, israndom=israndom)
    X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
    n_classes = Y_test.max()+1
    y_test = np_utils.to_categorical(Y_test, n_classes)
    y_test = np.squeeze(y_test)
 
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999,decay=1e-2,epsilon=1e-05)     
    
    #model = RNN(time_step=time_step,nb_features=nb_features,num_classes=num_classes)
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])    
    #print(model.summary())
    #model.fit([X_train], [y_train], epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True,validation_data=([X_test], [y_test]))
    #model.save('./0707/rnn_9_pcpu.h5')
    #model.save('./model2/rnn_9_pupc.h5')    
        
    #model = SSAN(time_step,nb_features,num_PC=num_PC,img_rows=w,img_cols=w,num_classes=num_classes)
    #model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #print(X_train.shape,XP_train.shape,y_train.shape)
    #print(X_test.shape,XP_test.shape,y_test.shape)
    #model.fit([X_train,XP_train],[y_train], epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True,validation_data=([X_test,XP_test],[y_test]))     
    #model.save('./vis_pkl/ssan_9_pcpu.h5')
    
    model = RESNET(num_PC=num_PC,img_rows=w,img_cols=w,num_classes=num_classes)
    #model = CNN(num_PC=num_PC,img_rows=w,img_cols=w,num_classes=num_classes)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([XP_train], [y_train], epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=True,validation_data=(XP_test, y_test)) #OVERFITTING
    model.save('./res_9_pcpu.h5')
    #model.save('./model/res_16_insa.h5')

    #model = load_model('./res_9_pupc.h5')
    #PredictLabel = model.predict([X_test,XP_test],verbose=0).argmax(axis=-1)
    #PredictLabel = model.predict([X_test],verbose=0).argmax(axis=-1)
    PredictLabel = model.predict([XP_test],verbose=0).argmax(axis=-1)
    correct_prediction = np.equal(PredictLabel, Y_test[:,0])
    accuracy = np.mean((correct_prediction).astype('float32'))
    print('test_accruacy:',accuracy)
        
    #PredictLabel = model.predict([X_train,XP_train],verbose=0).argmax(axis=-1)
    #PredictLabel = model.predict([X_train],verbose=0).argmax(axis=-1)
    PredictLabel = model.predict([XP_train],verbose=0).argmax(axis=-1)
    correct_prediction = np.equal(PredictLabel, Y_train[:,0])
    accuracy = np.mean((correct_prediction).astype('float32'))
    print('training_accruacy:',accuracy)     
    
