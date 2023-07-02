import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree
import pickle
from utils import*

parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 64)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-4)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()

# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

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
    # step 1: init dataset
    print("init dataset")
    WORD2VECPATH    = "./word2vec/word2vec_pcpu.npy"  
    class_vectors = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
    _,allClassVectors = zip(*class_vectors)
    #allClassVectors = np.asarray(allClassVectors)
    
    DATAPATH1       = "./vis_pkl/ssan_tr_pu.pkl"
    DATAPATH2       = "./vis_pkl/ssan_te_pc.pkl"
    (_,trainFeatures),(te_labels,unseenFeatures)=load_data(DATAPATH1,DATAPATH2)
    
    time_step,w,num_PC = 100,28,3
    dataID_tr,dataID_te = 1,2
    data = HyperspectralSamples(dataID=dataID_tr,timestep=time_step,w=w,num_PC=num_PC,israndom=True)
    X_train,XP_train,Y_train,label_train = data[1],data[3],data[5]-1,data[6]          
    data_te = HyperspectralSamples(dataID=dataID_te,timestep=time_step,w=w,num_PC=num_PC,israndom=True)
    X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
    trainLabels = np.squeeze(Y_train)
    unseenLabels = np.squeeze(Y_test)
        
    train_vectors = [instance[1] for instance in class_vectors if instance[0] in label_train]
    test_vectors = [instance[1] for instance in class_vectors if instance[0] in label_test]
    train_vectors = np.asarray(train_vectors)
    test_vectors = np.asarray(test_vectors)        
    # train set
    train_features=torch.from_numpy(trainFeatures)
    print('train_features.shape',train_features.shape)
    train_label=torch.from_numpy(trainLabels).unsqueeze(1)
    print('train_label.shape',train_label.shape)
    # attributes           
    all_attributes=np.asarray(allClassVectors) 
    attributes = torch.from_numpy(all_attributes)
    # test set
    test_features=torch.from_numpy(unseenFeatures)
    print('test_features.shape',test_features.shape)
    test_label=torch.from_numpy(unseenLabels).unsqueeze(1)
    print('test_label.shape',test_label.shape)
    test_id = np.unique(unseenLabels)
    testclasses_id = np.array(test_id)
    print('testclasses_id.shape',testclasses_id.shape)
    test_attributes = torch.from_numpy(test_vectors).float()
    print('test_attributes.shape',test_attributes.shape)           
    train_data = TensorDataset(train_features,train_label)
    
    # init network
    print("init networks")
    attribute_network = AttributeNetwork(300,512,1024)#(300,256,512)
    relation_network = RelationNetwork(2048,1024)#(1024,512)

    attribute_network.cuda(GPU)
    relation_network.cuda(GPU)

    attribute_network_optim = torch.optim.Adam(attribute_network.parameters(),lr=LEARNING_RATE,weight_decay=1e-5)
    attribute_network_scheduler = StepLR(attribute_network_optim,step_size=20000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=20000,gamma=0.5)
    
    print("training...")
    last1_accuracy,last2 = 0.0,0.0
    for episode in range(EPISODE):
        attribute_network_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
        batch_features,batch_labels = train_loader.__iter__().next()

        sample_labels = []
        for label in batch_labels.numpy():
            if label not in sample_labels:
                sample_labels.append(label)
        
        sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
        class_num = sample_attributes.shape[0]

        batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
        sample_features = attribute_network(Variable(sample_attributes).cuda(GPU)) #k*312

        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,2048)#(-1,1024)
        relations = relation_network(relation_pairs).view(-1,class_num)
        

        # re-build batch_labels according to sample_labels
        sample_labels = np.array(sample_labels)
        re_batch_labels = []
        for label in batch_labels.numpy():
            index = np.argwhere(sample_labels==label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)        
        # loss        
        #mse = nn.HingeEmbeddingLoss(margin=1, reduction='mean').cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1,1), 1)).cuda(GPU)
        mse = nn.MSELoss().cuda(GPU)
        loss = mse(relations,one_hot_labels)
        
        # update
        attribute_network.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        attribute_network_optim.step()
        relation_network_optim.step()
        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.item())
        if (episode+1)%2000 == 0:
            # test
            print("Testing...")            
            
            def compute_accuracy(test_features,test_label,test_id,test_attributes):                
                test_data = TensorDataset(test_features,test_label)
                test_batch = 128
                test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
                total_rewards = 0
                sample_labels = test_id
                sample_attributes = test_attributes
                class_num = sample_attributes.shape[0]
                test_size = test_features.shape[0]                
                print("class num:",class_num)
                predict_labels_total = []
                re_batch_labels_total = []
                pre_all_total = []                
                for batch_features,batch_labels in test_loader:
                    batch_size = batch_labels.shape[0]

                    batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
                    sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())

                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1)
                    batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
                    batch_features_ext = torch.transpose(batch_features_ext,0,1)

                    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,2048)#(-1,1024)
                    relations = relation_network(relation_pairs).view(-1,class_num)

                    re_batch_labels = []
                    for label in batch_labels.numpy():
                        index = np.argwhere(sample_labels==label)
                        re_batch_labels.append(index[0][0])
                    re_batch_labels = torch.LongTensor(re_batch_labels)

                    _,predict_labels = torch.max(relations.data,1)
                    predict_labels = predict_labels.cpu().numpy()
                    re_batch_labels = re_batch_labels.cpu().numpy()
                    _,pre_all = torch.sort(relations.data,dim=1,descending=True)
                    pre_all = np.asarray(pre_all.cpu().numpy())
                    predict_labels_total = np.append(predict_labels_total, predict_labels)
                    re_batch_labels_total = np.append(re_batch_labels_total, re_batch_labels)  
                    pre_all_total = np.append(pre_all_total,pre_all)              
                # compute averaged per class accuracy    
                predict_labels_total = np.array(predict_labels_total, dtype='int')
                re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
                pre_all_total = (np.array(pre_all_total,dtype='int')).reshape(re_batch_labels_total.shape[0],class_num)
                unique_labels = np.unique(re_batch_labels_total)
                top1_acc,top3_acc  = 0,0
                for l in unique_labels:
                    idx = np.nonzero(re_batch_labels_total == l)[0]
                    top1_acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])
                    t3=0
                    for i in range(len(re_batch_labels_total[idx])):
                        if re_batch_labels_total[idx][i] in pre_all_total[idx,:3][i]:
                            t3 = t3+1 
                    t3 = t3/ len(re_batch_labels_total[idx])       
                    top3_acc += t3
                top1_acc = top1_acc / unique_labels.shape[0]
                top3_acc = top3_acc/ unique_labels.shape[0]
                return top1_acc,top3_acc            
            top1_acc,top3_acc= compute_accuracy(test_features,test_label,test_id,test_attributes)            
            print('zsl:', top1_acc,top3_acc)                        
            if top1_acc > last1_accuracy:
                last1_accuracy, last2 = top1_acc,top3_acc
            print('lase_accuracy',last1_accuracy,last2)
                # save networks
                #torch.save(attribute_network.state_dict(),"./model/zsl_an.pkl")
                #torch.save(relation_network.state_dict(),"./model/zsl_rn.pkl")
                #print("save networks for episode:",episode)
                

if __name__ == '__main__':

    main()

   