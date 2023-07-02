from scipy import io
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import kNN
import pickle
from utils import*

def compute_accuracy(test_att, test_visual, test_id, test_label):
	# test_att: [2993, 312]
	# test viaual: [2993, 1024]
	# test_id: att2label [50]
	# test_label: x2label [2993]
	test_att = Variable(torch.from_numpy(test_att).float().cuda())
	att_pred = forward(test_att)
	outpred = [0] * test_visual.shape[0]
	test_label = test_label.astype("float32")

	# att_pre [50, 1024],
	# test_visual: [2993, 1024]
	# test_id : [50]
  #pred_all = np.zeros((test_visual.shape[0],test_id.shape[0]))
	top3 = 0
	for i in range(test_visual.shape[0]):
		outputLabel = kNN.kNNClassify(test_visual[i, :], att_pred.cpu().data.numpy(), test_id, 1)
		outpred[i] = outputLabel
		sortedDistIndices = kNN.kNNtopk(test_visual[i, :], att_pred.cpu().data.numpy(), test_id)
		#pred_all[i] = sortedDistIndices
		if test_label[i] in sortedDistIndices[:3]:
				top3 +=1   
	top3_acc = top3 / test_visual.shape[0]
	outpred = np.array(outpred)	
	acc = np.equal(outpred, test_label).mean()
	return acc,top3_acc

def data_iterator():
	""" A simple data iterator """
	batch_idx = 0
	while True:
		# shuffle labels and features
		idxs = np.arange(0, len(train_x))
		np.random.shuffle(idxs)
		shuf_visual = train_x[idxs]
		shuf_att = train_att[idxs]
		batch_size = 100

		for batch_idx in range(0, len(train_x), batch_size):
			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
			visual_batch = visual_batch.astype("float32")
			att_batch = shuf_att[batch_idx:batch_idx + batch_size]

			att_batch = Variable(torch.from_numpy(att_batch).float().cuda())
			visual_batch = Variable(torch.from_numpy(visual_batch).float().cuda())
			yield att_batch, visual_batch

def load_data(DATAPATH1,DATAPATH2):
	"""read data, create datasets"""
	# READ DATA
	f1 = open(DATAPATH1,'rb')
	data1 = pickle.load(f1)
	#pickle.dump(data1, open(DATAPATH1,"wb"), protocol=2)
	f2 = open(DATAPATH2,'rb')
	data2 = pickle.load(f2)
	# FORM X_TRAIN AND Y_TRAIN
	tr_labels,tr_feats    = zip(*data1)
	tr_labels,tr_feats    = (np.asarray(tr_labels)), np.squeeze(np.asarray(tr_feats))
	# FORM X_ZSL AND Y_ZSL
	te_labels,te_feats = zip(*data2)
	te_labels,te_feats = (np.asarray(te_labels)), np.squeeze(np.asarray(te_feats))
	return (tr_labels,tr_feats), (te_labels,te_feats) 
 
def labels_2_embeddings(labels,classname,class_vectors):    
	labels = np.squeeze(labels)
	labels = labels.tolist()
	label_embeddings = np.zeros((len(labels),300))
	for i,label in enumerate(labels):
		for j in range(len(class_vectors)):
			if class_vectors[j][0] == classname[label] :
				label_embeddings[i] = class_vectors[j][1]  
	return label_embeddings


time_step,w,num_PC = 10,28,3
dataID_tr,dataID_te = 1,2
###############train################
data = HyperspectralSamples(dataID=dataID_tr, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
X_train,XP_train,Y_train,label_train = data[1],data[3],data[5]-1,data[6]       
#nb_classes = Y_train.max()+1
#y_train = np.squeeze(np_utils.to_categorical(Y_train, nb_classes))        
###############test#################    
data_te = HyperspectralSamples(dataID=dataID_te, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
WORD2VECPATH    = "./word2vec/word2vec_pcpu.npy"  
class_vectors = sorted(np.load(WORD2VECPATH,allow_pickle=True), key=lambda x: x[0])
DATAPATH1       = "./vis_pkl/res_tr_pu.pkl"
DATAPATH2       = "./vis_pkl/res_te_pc.pkl"
(_,train_feats),(_,test_feats)=load_data(DATAPATH1,DATAPATH2)
train_label_embeddings = labels_2_embeddings(Y_train,label_train,class_vectors)
test_class_vectors = [instance[1] for instance in class_vectors if instance[0] in label_test]
test_class_vectors = np.asarray(test_class_vectors)

train_att = train_label_embeddings
print(train_att.shape)
train_x = train_feats
print(train_x.shape)
test_x = test_feats
print(test_x.shape)
test_x2label = np.squeeze(Y_test)
print(test_x2label.shape)
test_att2label = np.squeeze(np.array(np.unique(Y_test)))
print(test_att2label.shape)
test_att = test_class_vectors
print(test_att.shape)

w1 = Variable(torch.FloatTensor(300, 256).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(256).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(256, 512).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(512).cuda(), requires_grad=True)

# must initialize!
w1.data.normal_(0, 0.3)
w2.data.normal_(0, 0.03)
b1.data.fill_(0)
b2.data.fill_(0)


def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)
	return a2

def getloss(pred, x):
	loss = torch.pow(x - pred, 2).sum()
	loss /= x.size(0)
	return loss

optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=1e-5, weight_decay=1e-2)
# # Run
iter_ = data_iterator()
last_top1,last_top3 = 0.,0.
for i in range(100000):
	att_batch_val, visual_batch_val = next(iter_)

	pred = forward(att_batch_val)
	loss = getloss(pred, visual_batch_val)

	optimizer.zero_grad()
	loss.backward() 
	# gradient clip makes it converge much faster!
	torch.nn.utils.clip_grad_norm([w1, b1, w2, b2], 1)
	optimizer.step()
	if i % 2000 == 0:
		top1,top3 = compute_accuracy(test_att, test_x, test_att2label, test_x2label)
		print(i,top1,top3)
		if top1> last_top1 :
				last_top1 = top1
				last_top3 = top3
print("last:",last_top1,last_top3)