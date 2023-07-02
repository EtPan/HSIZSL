import scipy.io as sio  
import numpy as np  
import pickle
import tensorflow as tf
from utils import*
from sklearn.neighbors import KDTree

BATCH_SIZE = 128
EPOCH_BOUND = 50
EARLY_STOP_CHECK_EPOCH = 20
MARGIN = 0.1
EPOCH_PER_DECAY = 10
STORED_PATH = "./saved_model/devise.ckpt"         
LOGS_PATH = "./logs/devise"

def labels_2_embeddings(labels,classname,class_vectors):    
	labels = np.squeeze(labels)
	labels = labels.tolist()
	label_embeddings = np.zeros((len(labels),300))
	for i,label in enumerate(labels):
		for j in range(len(class_vectors)):
			if class_vectors[j][0] == classname[label] :
				label_embeddings[i] = class_vectors[j][1]  
	return label_embeddings

def get_classes_text_embedding(self, embedding_dim, classes):
	classes_text_embedding = []
	for class_label in classes:
		if "_" in class_label:
			word_len = 0
			embedding = [0.0 for x in range(embedding_dim)]
			for word in class_label.split("_"):
				word_len += 1
				embedding += self.model[word]
			classes_text_embedding.append(embedding/word_len)
		else: 
			classes_text_embedding.append(self.model[class_label])
	return np.array(classes_text_embedding, dtype=np.float32)

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
	time_step,w,num_PC = 10,28,3
	dataID_tr,dataID_te = 2,1
	###############train################
	data = HyperspectralSamples(dataID=dataID_tr, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
	X_train,XP_train,Y_train,label_train = data[1],data[3],data[5]-1,data[6]
	train_labels = Y_train
	#nb_classes = Y_train.max()+1
	#y_train = np.squeeze(np_utils.to_categorical(Y_train, nb_classes))       
	###############test#################    
	data_te = HyperspectralSamples(dataID=dataID_te, timestep=time_step, w=w, num_PC=num_PC, israndom=True)
	X_test,XP_test,Y_test,label_test = data_te[1],data_te[3],data_te[5]-1,data_te[6]
	test_labels = Y_test

	WORD2VECPATH    = "./word2vec/word2vec_pcpu.npy"  
	class_vectors = sorted(np.load(WORD2VECPATH), key=lambda x: x[0])
	_,classes_text_embedding = zip(*class_vectors)

	DATAPATH1       = "./vis_pkl/ssun_tr_pc.pkl"
	DATAPATH2       = "./vis_pkl/ssan_te_pu.pkl"
	(_,train_feats),(test_names,test_feats)=load_data(DATAPATH1,DATAPATH2)
	train_label_embeddings = labels_2_embeddings(Y_train,label_train,class_vectors)
	test_class_vectors = [instance[1] for instance in class_vectors if instance[0] in label_test]
	test_class_vectors = np.asarray(test_class_vectors)
	test_label_embeddings = labels_2_embeddings(Y_test,label_test,class_vectors)
 
	x  = tf.placeholder(tf.float32,[None,train_feats.shape[1]],name = 'x')
	y  = tf.placeholder(tf.int64,[None,1],name = 'y')
	yy = tf.placeholder(tf.float32,[None,train_label_embeddings.shape[1]],name = 'yy')

	with tf.name_scope('transform'):
		x_tr = tf.layers.dense(inputs = x,units=300,name='transform')
		tf.summary.histogram('x_tr', x_tr)
	training_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="transform")
	
	with tf.name_scope('devise_loss'):
		loss = tf.constant(0.0)
		predic_true_distance = tf.reduce_sum(tf.multiply(yy, x_tr), axis=1, keep_dims=True)
		for j in range(len(classes_text_embedding)):
			loss = tf.add(loss, tf.maximum(0.0, (MARGIN - predic_true_distance 
                                    + tf.reduce_sum(tf.multiply(classes_text_embedding[j], x_tr),axis=1, keep_dims=True))))
		loss = tf.subtract(loss, MARGIN)
		loss = tf.reduce_sum(loss)
		loss = tf.div(loss, BATCH_SIZE)
		tf.summary.scalar('loss', loss)
	print("loss defined")
  
	decay_steps = int(BATCH_SIZE * EPOCH_PER_DECAY)
	global_step = tf.train.get_or_create_global_step()
	learning_rate = tf.train.exponential_decay(
        learning_rate=0.0001, #initial learning rate
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=0.96,
        staircase=True,
        name='ExponentialDecayLearningRate')
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='GD_Optimizer')
	train_op = optimizer.minimize(loss, name='train_op', var_list=training_vars)
	ema = tf.train.ExponentialMovingAverage(decay=0.9999)
	with tf.control_dependencies([train_op]):
		train_op = ema.apply(training_vars)

		
	print("########## Start training ##########")
	# randomize dataset
	indices = np.random.permutation(train_feats.shape[0])
	train_feats,train_labels,train_vectors = np.array(train_feats[indices,:]),np.array(train_labels[indices,:]),np.array(train_label_embeddings[indices,:]) 
	
	sess = tf.Session()
	init = tf.global_variables_initializer()
	# init saver to save model
	saver = tf.train.Saver()
	# init weights
	sess.run(init)
	for var in training_vars:
		sess.run(var.initializer)
	# visualize data
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)

	optimizer,epoch_bound,stop_threshold, batch_size = train_op,EPOCH_BOUND, EARLY_STOP_CHECK_EPOCH, BATCH_SIZE	
	X_train, y_train, yy_train =train_feats,train_labels,train_vectors
	early_stop = 0
	best_loss,best_acc = np.infty,0.0
	for epoch in range(epoch_bound):
		# randomize training set
		indices_training = np.random.permutation(X_train.shape[0])
		X_train, yy_train = X_train[indices_training,:], yy_train[indices_training,:]
		# split training set into multiple mini-batches and start training        
		total_batches = int(X_train.shape[0] / batch_size)
		cur_loss=0.0
		for batch in range(total_batches):
			if batch == total_batches - 1:
				sess.run(optimizer, feed_dict={x: X_train[batch*batch_size:], 
                                       y: y_train[batch*batch_size:],
                                       yy: yy_train[batch*batch_size:]})
				c_loss,summary = sess.run([loss,merged], feed_dict={x: X_train[batch*batch_size:],
                                                      y: y_train[batch*batch_size:],
                                                      yy: yy_train[batch*batch_size:]})
				writer.add_summary(summary, epoch + (batch/total_batches))
				cur_loss += c_loss
			else:
				sess.run(optimizer, feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size], 
                                          y: y_train[batch*batch_size : (batch+1)*batch_size], 
                                          yy: yy_train[batch*batch_size : (batch+1)*batch_size]})
				c_loss,summary = sess.run([loss,merged], feed_dict={x: X_train[batch*batch_size : (batch+1)*batch_size],
                                                      y: y_train[batch*batch_size : (batch+1)*batch_size],
                                                      yy: yy_train[batch*batch_size : (batch+1)*batch_size]})
				writer.add_summary(summary, epoch + (batch/total_batches))
				cur_loss += c_loss
		cur_loss /= total_batches  
		if best_loss > cur_loss:
			early_stop = 0
			best_loss = cur_loss
			save_path = saver.save(sess, STORED_PATH)
		else:
			early_stop += 1
		print('\tEpoch: ', epoch, '\tBest loss: ', best_loss, '\tCurrent loss: ', cur_loss)
		if early_stop == stop_threshold:
			break  
	print("training with all the inputs, loss:", best_loss)
	#sess.close()
 
	########## Evaluate ##########
	# Evaluate the model and print results
	print("########## Start evaluating ##########")
	X_test,y_test,yy_test = test_feats,test_labels,test_label_embeddings
	#sess = tf.Session()
	# restore the precious best model
	#saver = tf.train.Saver()
	#saver.restore(sess, STORED_PATH)
	predict_embeddings= sess.run(x_tr, feed_dict={x:X_test,y:y_test,yy:yy_test})

	sess.close()
	tree        = KDTree(test_class_vectors)
	pred_zsl    = predict_embeddings
	top3, top1 = 0, 0
	for i, pred in enumerate(pred_zsl):
		pred            = np.expand_dims(pred, axis=0)
		dist_5, index_5 = tree.query(pred, k=5)
		pred_labels     = [label_test[index] for index in index_5[0]]
		true_label      = test_names[i] 
		if true_label in pred_labels[:3]:
			top3 += 1
		if true_label in pred_labels[0]:
			top1 += 1
	print("ZERO SHOT LEARNING SCORE")
	print("-> Top-3 Accuracy: %.4f" % (top3 / float(len(test_labels))))
	print("-> Top-1 Accuracy: %.4f" % (top1 / float(len(test_labels))))

if __name__ == '__main__':  

	main()