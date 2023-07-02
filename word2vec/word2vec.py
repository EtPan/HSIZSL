import os
from smart_open import smart_open
import gensim
import numpy as np
import tensorflow as tf

def clean_text(x):
    x = str(x)
    x.replace('/>', ' /> ')
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '/_':
        x = x.replace(punct, ' ')
    return x

class MyLabel(object):
    def __init__(self, dirname):
        self.dirname = dirname
        lines = [clean_text(line) for line in smart_open(os.path.join(self.dirname), 'r') ]
        lines = list(map(lambda x: x.split(), lines))      
        self.size = len(lines)
        self.voc = [(item[0], item[1]) for item in zip(range(self.size),lines)]
        self.voc = dict(self.voc)
        self.label = [str.strip(line) for line in smart_open(os.path.join(self.dirname), 'r')]

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return 0

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_word2vec(fname, vocab,vec_file):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    vector= {}
    label = {}
    voc_vec = ()
    label_vec=[]
    for index, words in vocab.voc.items():
        vector[index] = np.mean([model[word] for word in words], axis=0) 
        label [index] = vocab.label[index]
        voc_vec = (label[index],vector[index])
        label_vec.append(voc_vec)
    np.save(vec_file,label_vec)
    print(vector.shape, label_vec.shape)
    return vector,label_vec

from sklearn.decomposition import PCA
from matplotlib import pyplot
def pca_2d(vec,):
    
    pca = PCA(n_components=2)
    result = pca.fit_transform(vec)
    pyplot.scatter(result[:,0],result[:,1])
    pyplot.scatter


if __name__=="__main__":
    w2v_file = "./model/GoogleNews-vectors-negative300.bin"
    #glove_file = './model/glove.840B.300d.txt'    
    #from gensim.scripts.glove2word2vec import glove2word2vec
    #glove_temp ='./model/glove.840B.300d_temp.txt'
    #glove2word2vec(glove_file,glove_temp)
    
    labellist = 'labellist_pcpu.txt'
    vec_file = './word2vec_pcpu_temp.npy'    
    vocab = MyLabel(labellist) # a memory-friendly iterator
    w2v,label_vec= load_word2vec(w2v_file,vocab,vec_file)
    print('ok')
    
  
 
    '''
    labellist = 'labellist_insa.txt'
    vec_file = './word2vec_insa.npy'    
    vocab = MyLabel(labellist) # a memory-friendly iterator
    w2v,label_vec= load_word2vec(w2v_file,vocab,vec_file)    
    print('ok')
    '''
    '''
    from tensorflow.contrib.tensorboard.plugins import projector
    summary_writer = tf.summary.FileWriter(¡®checkpoints¡¯, sess.graph)
    
    config = projector.ProjectorConfig()
    
    embedding_conf = config.embeddings.add()
    # embedding_conf.tensor_name = ¡®embedding:0¡¯
    embedding_conf.metadata_path = os.path.join(¡®checkpoints¡¯, ¡®metadata.tsv¡¯)
    projector.visualize_embeddings(summary_writer, config)
    '''
    '''    
    from gensim.models import Word2Vec
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
    model = Word2Vec(sentences,window=5, min_count=1)
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
	    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()
    ''' 