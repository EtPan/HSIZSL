import random
random.seed(456)
import numpy as np
np.random.seed(456)
import torch
torch.manual_seed(456)

from torch.utils.data import TensorDataset, DataLoader
from easydict import EasyDict as edict
import pickle
from utils import*

class Network(torch.nn.Module):
    """ Zero-Shot model """
    def __init__(self, feature_dim, vector_dim):
        super(Network, self).__init__()
        self.wHidden1   = torch.nn.Linear(feature_dim, vector_dim)
    def forward(self, imageFeatures, classVectors):
        imageFeatures   = self.wHidden1(imageFeatures)
        out             = torch.matmul(imageFeatures, torch.t(classVectors))
        return out

def evaluate(model, x, y, vec):
    """ Normalized Zero-Shot Evaluation Method """
    classIndices    = np.unique(y.numpy())
    n_class         = len(classIndices)
    t_acc ,t3_acc   = 0.,0.
    y_preds         = model(x, vec)
    for index in classIndices:
        sampleIndices   = [i for i, _y in enumerate(y) if _y==index]
        n_sample        = len(sampleIndices)
        y_sample_preds  = torch.argmax(y_preds[sampleIndices], dim=1)
        y_samples       = y[sampleIndices]
        sampleScore     = torch.sum(y_sample_preds == y_samples).item()
        sampleAcc       = sampleScore / n_sample
        t_acc           += sampleAcc
        _,pre_all = torch.sort(y_preds[sampleIndices],dim=1,descending=True)
        t3 = 0
        for i in range(y_samples.shape[0]):
            if y_samples[i] in pre_all[i,:3] :
                t3  +=1
        t3   =  t3 /y_samples.shape[0] 
        t3_acc +=t3    
    acc = t_acc / n_class
    #t3_acc = t3_acc / y_samples.shape[0]
    t3_acc = t3_acc / n_class
    return acc,t3_acc

def load_data(DATAPATH1,DATAPATH2):
	"""read data, create datasets"""
	# READ DATA
	f1 = open(DATAPATH1,'rb')
	data1 = pickle.load(f1)
	f2 = open(DATAPATH2,'rb')
	data2 = pickle.load(f2)
	# FORM X_TRAIN AND Y_TRAIN
	tr_labels,tr_feats    = zip(*data1)
	tr_labels,tr_feats    = (np.asarray(tr_labels)), np.squeeze(np.asarray(tr_feats))
	# FORM X_ZSL AND Y_ZSL      
	te_labels,te_feats = zip(*data2)
	te_labels,te_feats = (np.asarray(te_labels)), np.squeeze(np.asarray(te_feats))
	return (tr_labels,tr_feats), (te_labels,te_feats) 
    
def main():
    print('#####    TEST PHASE    #####')
    # ------------------------------------------------------------------------------- #
    #load data    
    WORD2VECPATH    = "./word2vec/word2vec_pcpu.npy"  
    class_vectors = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
    _,allClassVectors = zip(*class_vectors)
    allClassVectors = np.asarray(allClassVectors)
    
    DATAPATH1       = "./vis_pkl/ssan_tr_pc.pkl"
    DATAPATH2       = "./vis_pkl/ssan_te_pu.pkl"
    (_,trainvalFeatures),(_,unseenFeatures)=load_data(DATAPATH1,DATAPATH2)
    
    time_step,w,num_PC = 10,28,3
    dataID_tr,dataID_te = 2,1
    data = HyperspectralSamples(dataID=dataID_tr,timestep=time_step,w=w,num_PC=num_PC,israndom=True)
    X_train,XP_train,Y_train,label_train = data[1],data[3],data[5]-1,data[6]          
    data_te = HyperspectralSamples(dataID=dataID_te,timestep=time_step,w=w,num_PC=num_PC,israndom=True)
    X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
    trainvalLabels = np.squeeze(Y_train)
    unseenLabels =np.squeeze(Y_test)
    
    print("##" * 25)
    print("All Class Vectors            : ", allClassVectors.shape)
    print("TrainVal Features            : ", trainvalFeatures.shape)
    print("TrainVal Labels              : ", trainvalLabels.shape)
    print("Unseen Features              : ", unseenFeatures.shape)
    print("Unseen Labels                : ", unseenLabels.shape)
    print("##" * 25)
    # ------------------------------------------------------------------------------- #
    # get data information
    n_class, attr_dim   = allClassVectors.shape
    n_train, feat_dim   = trainvalFeatures.shape
    n_unseen, _         = unseenFeatures.shape
    print("##" * 25)
    print("Number of Classes            : ", n_class)
    print("Number of Train samples      : ", n_train)
    print("Number of Unseen samples     : ", n_unseen)
    print("Attribute Dim                : ", attr_dim)
    print("Feature Dim                  : ", feat_dim)
    print("##" * 25)
    # ------------------------------------------------------------------------------- #
    
    trainval_vectors = [instance[1] for instance in class_vectors if instance[0] in label_train]
    test_vectors = [instance[1] for instance in class_vectors if instance[0] in label_test]
    trainval_vectors = np.asarray(trainval_vectors)
    test_vectors = np.asarray(test_vectors)
    print(trainval_vectors.shape)
    print(test_vectors.shape)
    # ------------------------------------------------------------------------------- #  
    # set network hyper-parameters
    n_epoch     = 30
    batch_size  = 64
    lr          = 1e-3
    # set network architecture, optimizer and loss function
    model       = Network(feature_dim=feat_dim, vector_dim=attr_dim)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)   # <-- Optimizer
    criterion   = torch.nn.CrossEntropyLoss(reduction='sum')    # <-- Loss Function
    # ------------------------------------------------------------------------------ #
    # convert data from numpy arrays to pytorch tensors
    x_trainvalFeatures  = torch.from_numpy(trainvalFeatures).float()
    y_trainvalLabels    = torch.from_numpy(trainvalLabels).long()

    x_unseenFeatures    = torch.from_numpy(unseenFeatures).float()
    y_unseenLabels      = torch.from_numpy(unseenLabels).long()

    seenVectors         = torch.from_numpy(trainval_vectors).float()
    unseenVectors       = torch.from_numpy(test_vectors).float()
    print("##" * 25)
    print("Seen Vector shape            : ", tuple(seenVectors.size()))
    print("Unseen Vector shape          : ", tuple(unseenVectors.size()))
    print("##" * 25)
    # initialize data loader
    trainvalData    = TensorDataset(x_trainvalFeatures, y_trainvalLabels)
    trainvalLoader  = DataLoader(trainvalData, batch_size=batch_size, shuffle=True)    
    # **************************************************** #
    #             ATTRIBUTE LABEL EMBEDDING     
    # **************************************************** #
    max_zslAcc      = float('-inf')
    # -------------------- #
    #        TRAINING      #
    # -------------------- #
    for epochID in range(n_epoch):

        model.train()       # <-- Train Mode On
        running_trainval_loss = 0.
        for x, y in trainvalLoader:
            y_out           = model(x, seenVectors)
            trainval_loss   = criterion(y_out, y)
            optimizer.zero_grad()       # <-- set gradients to zero
            trainval_loss.backward()    # <-- calculate gradients
            optimizer.step()            # <-- update weights
            running_trainval_loss += trainval_loss.item()
        # ---------------------- #
        #       PRINT LOSS       #
        # ---------------------- #
        print("%s\tTrain Loss: %s" % (str(epochID + 1), str(running_trainval_loss / n_train)))
        if (epochID + 1) % 1 == 0:
            model.eval()  # <-- Evaluation Mode On
            print("##" * 25)
            # TRAIN ACCURACY
            y_out       = model(x_trainvalFeatures, seenVectors)
            y_out       = torch.argmax(y_out, dim=1)
            trainScore  = torch.sum(y_out == y_trainvalLabels).item()
            trainAcc    = trainScore / n_train
            print("Train acc              : %s" % str(trainAcc))
            # ZERO-SHOT ACCURACY
            zslAcc ,t3_acc     = evaluate( model   = model,
                                    x       = x_unseenFeatures,
                                    y       = y_unseenLabels,
                                    vec     = unseenVectors)
            print("Zero-Shot acc          : %s,%s" % (str(zslAcc),str(t3_acc)))
            if zslAcc > max_zslAcc:
                max_zslAcc      = zslAcc
                #torch.save(model, 'model/ale.pt')
                #print("ALE is saved.")
    print("Zsl Acc: %.6s,%.6s"%(str(max_zslAcc),str(t3_acc)))
    return

if __name__ == '__main__':
    main()