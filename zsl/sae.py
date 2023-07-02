import numpy as np
import scipy
import scipy.io
import argparse
import pickle
from utils import*


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ld', type=float, default=500000) # lambda
	return parser.parse_args()

def normalizeFeature(x):
	# x = d x N dims (d: feature dimension, N: the number of features)
	x = x + 1e-10 # for avoid RuntimeWarning: invalid value encountered in divide
	feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	feat = x / feature_norm[:, np.newaxis]
	return feat

def SAE(x, s, ld):
	# SAE is Semantic Autoencoder
	# INPUTS:
	# 	x: d x N data matrix
	#	s: k x N semantic matrix
	#	ld: lambda for regularization parameter
	#
	# OUTPUT:
	#	w: kxd projection matrix

	A = np.dot(s, s.transpose())
	B = ld * np.dot(x, x.transpose())
	C = (1+ld) * np.dot(s, x.transpose())
	w = scipy.linalg.solve_sylvester(A,B,C)
	return w

def distCosine(x, y):
	xx = np.sum(x**2, axis=1)**0.5
	x = x / xx[:, np.newaxis]
	yy = np.sum(y**2, axis=1)**0.5
	y = y / yy[:, np.newaxis]
	dist = 1 - np.dot(x, y.transpose())
	return dist

def zsl_acc(semantic_predicted, semantic_gt, opts):
	# zsl_acc calculates zero-shot classification accruacy
	#
	# INPUTS:
	#	semantic_prediced: predicted semantic labels
	# 	semantic_gt: ground truth semantic labels
	# 	opts: other parameters
	#
	# OUTPUT:
	# 	zsl_accuracy: zero-shot classification accuracy (per-sample)

	dist = 1 - distCosine(semantic_predicted, normalizeFeature(semantic_gt.transpose()).transpose())
	y_hit_k = np.zeros((dist.shape[0], opts.HITK))
	for idx in range(0, dist.shape[0]):
		sorted_id = sorted(range(len(dist[idx,:])), key=lambda k: dist[idx,:][k], reverse=True)
		y_hit_k[idx,:] = opts.test_classes_id[sorted_id[0:opts.HITK]]
		
	n = 0
	for idx in range(0, dist.shape[0]):
		if opts.test_labels[idx] in y_hit_k[idx,:]:
			n = n + 1
	zsl_accuracy = float(n) / dist.shape[0] * 100
	return zsl_accuracy, y_hit_k

def labels_2_embeddings(labels,classname,class_vectors):    
	labels = np.squeeze(labels)
	labels = labels.tolist()
	label_embeddings = np.zeros((len(labels),300))
	for i,label in enumerate(labels):
		for j in range(len(class_vectors)):
			if class_vectors[j][0] == classname[label] :
				label_embeddings[i] = class_vectors[j][1]  
	return label_embeddings

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
	# for AwA dataset: Perfectly works.
	opts = parse_args()

	time_step,w,num_PC = 10,28,3
	dataID_tr,dataID_te = 2,1
	###############train################
	data = HyperspectralSamples(dataID=dataID_tr, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
	X_train,XP_train,Y_train,label_train = data[1],data[3],data[5]-1,data[6]       
	#nb_classes = Y_train.max()+1
	#y_train = np.squeeze(np_utils.to_categorical(Y_train, nb_classes))        
	###############test#################    
	data_te = HyperspectralSamples(dataID=dataID_te, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
	X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
	print(Y_train.shape,Y_test.shape)
           
	opts.test_labels = Y_test
	opts.test_classes_id = np.asarray(np.unique(Y_test))
	WORD2VECPATH    = "./word2vec/word2vec_pcpu.npy"  
	class_vectors = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])

	DATAPATH1       = "./vis_pkl/ssan_tr_pc.pkl"
	DATAPATH2       = "./vis_pkl/ssan_te_pu.pkl"
	(_,train_feats),(_,test_feats)=load_data(DATAPATH1,DATAPATH2)
	train_label_embeddings = labels_2_embeddings(Y_train,label_train,class_vectors)
	test_class_vectors = [instance[1] for instance in class_vectors if instance[0] in label_test]
	test_class_vectors = np.asarray(test_class_vectors)
	#test_label_embeddings  = labels_2_embeddings(Y_test,label_test,class_vectors)
	
	##### Normalize the data
	train_feats = normalizeFeature(train_feats.transpose()).transpose() 
	##### Training
	# SAE
	W = SAE(train_feats.transpose(), train_label_embeddings.transpose(), opts.ld) 
	##### Test
	opts.HITK = 3
	# [F --> S], projecting data from feature space to semantic space: 84.68% for AwA dataset
	semantic_predicted = np.dot(test_feats, normalizeFeature(W).transpose())
	[zsl_accuracy, y_hit_k] = zsl_acc(semantic_predicted, test_class_vectors, opts)
	print('[1] zsl accuracy for AwA dataset [F >>> S]: {:.2f}%'.format(zsl_accuracy))

	# [S --> F], projecting from semantic to visual space: 84.00% for AwA dataset
	test_predicted = np.dot(normalizeFeature(test_class_vectors.transpose()).transpose(), normalizeFeature(W))
	[zsl_accuracy, y_hit_k] = zsl_acc(test_feats, test_predicted, opts)
	print('[2] zsl accuracy for AwA dataset [S >>> F]: {:.2f}%'.format(zsl_accuracy))
	
if __name__ == '__main__':
  
	main()