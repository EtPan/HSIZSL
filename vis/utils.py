import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
    
def PCANorm(X,num_PC):
    mu = np.mean(X,0)
    X_norm = X-mu    
    Sigma = np.cov(X_norm.T)
    [U, S, V] = np.linalg.svd(Sigma)   
    XPCANorm = np.dot(X_norm,U[:,0:num_PC])
    return XPCANorm

def MirrowCut(X,hw):
    #X  size: row * column * num_feature

    [row,col,n_feature] = X.shape
    X_extension = np.zeros((3*row,3*col,n_feature))
    
    for i in range(0,n_feature):
        lr = np.fliplr(X[:,:,i])
        ud = np.flipud(X[:,:,i])
        lrud = np.fliplr(ud)
        
        l1 = np.concatenate((lrud,ud,lrud),axis=1)
        l2 = np.concatenate((lr,X[:,:,i],lr),axis=1)
        l3 = np.concatenate((lrud,ud,lrud),axis=1)
        
        X_extension[:,:,i] = np.concatenate((l1,l2,l3),axis=0)
    
    X_extension = X_extension[row-hw:2*row+hw,col-hw:2*col+hw,:]
    
    return X_extension

def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Indian Pines
    #ID=6:KSC
    num_class = labels.max()+1
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[0,0,0],
                            [216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    elif imageID == 3:
        row = 145
        col = 145
        palette = np.array([[255,0,0],
                            [0,255,0],
                            [0,0,255],
                            [255,255,0],
                            [0,255,255],
                            [255,0,255],
                            [176,48,96],
                            [46,139,87],
                            [160,32,240],
                            [255,127,80],
                            [127,255,212],
                            [218,112,214],
                            [160,82,45],
                            [127,255,0],
                            [216,191,216],
                            [238,0,0]])
        palette = palette*1.0/255
    elif imageID == 2:
        row = 1096
        col = 715
        palette = np.array([[0,0,0],[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    
    X_result = np.zeros((labels.shape[0],3))
    for i in range(0,num_class):
        X_result[np.where(labels==i),0] = palette[i,0]
        X_result[np.where(labels==i),1] = palette[i,1]
        X_result[np.where(labels==i),2] = palette[i,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    #plt.axis ( "off" ) 
    #plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA

def HyperspectralSamples(dataID=1, timestep=4, w=24, num_PC=3, israndom=False):   
    #dataID=1:Pavia University
    #dataID=2:Pavia Center
    #dataID=3:Salinas
    #dataID=4:Indian_pines
    if dataID==1:
        data = sio.loadmat('./Datasets/paviau/PaviaU.mat')
        X = data['paviaU']    
        data = sio.loadmat('./Datasets/paviau/PaviaU_gt.mat')
        Y = data['paviaU_gt']
        label = ['Asphalt','Meadows','Gravel','Trees','Painted metal sheets','Bare Soil','Bitumen','Self-Blocking Bricks','Shadows']
    elif dataID==2:
        data = sio.loadmat('./Datasets/paviac/Pavia.mat')
        X = data['pavia']    
        data = sio.loadmat('./Datasets/paviac/Pavia_gt.mat')
        Y = data['pavia_gt']
        label = ['Water','Trees','Asphalt','Self-Blocking Bricks','Bitumen','Tiles','Shadows','Meadows','Bare Soil']    
    if dataID==3:
        data = sio.loadmat('./Datasets/Salinas/Salinas_corrected_200.mat')
        X = data['salinas_corrected_200']    
        data = sio.loadmat('./Datasets/Datasets/Salinas/Salinas_gt.mat')
        Y = data['salinas_gt']
        label = ['Broccoli_green_weeds_1','Broccoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth','Stubble','Celery','Grapes_untrained','Soil_Vinyard_develop','Corn_green_weeds','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical_trellis']
    elif dataID==4:
        data = sio.loadmat('./Datasets/Indian_pines/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']    
        data = sio.loadmat('./Datasets/Indian_pines/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        label = ['Alfalfa','Corn-no-till','Corn-min-till','Corn','Grass-pasture','Grass-trees','Grass-pasture-mowed','Hay-windrow','Oats','Soybean-no-till','Soybean-min-till','Soybean-clean','Wheat','Woods','Buildings-Grass-Trees-Drives','Stone-Steel-Towers']  
    
            
    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(row*col, n_feature)    
    Y = Y.reshape(row*col, 1)    
    n_class = Y.max()
    nb_features = int(n_feature/timestep)                    
    X_PCA = featureNormalize(PCANorm(X,num_PC),2)    
    X = featureNormalize(X,1)
           
    hw = int(w/2)
    X_PCAMirrow = MirrowCut(X_PCA.reshape(row,col,num_PC),hw)    
    XP = np.zeros((K,w,w,num_PC))    
    for i in range(1,K+1):
        index_row = int(np.ceil(i*1.0/col))
        index_col = i - (index_row-1)*col + hw -1 
        index_row += hw -1
        patch = X_PCAMirrow[(index_row-hw):(index_row+hw),(index_col-hw):(index_col+hw),:]
        XP[i-1,:,:,:] = patch    

    
    if israndom==True:
        randomArray = list()
        for i in range(1,n_class+1):
            index = np.where(Y==i)[0]
            n_data = index.shape[0]
            randomArray.append(np.random.permutation(n_data))      

    train_num_array = [] 
    for i in range(1, n_class+1):
        indices = [j for j, x in enumerate(Y.ravel().tolist()) if x == i]
        train_num_i = int(0.2*len(indices))
        train_num_array.append(train_num_i)
    
    print(train_num_array)
    flag1=0
    flag2=0
    train_num_all = sum(train_num_array)        
    X_train = np.zeros((train_num_all,timestep,nb_features))
    XP_train = np.zeros((train_num_all,w,w,num_PC))  
    Y_train = np.zeros((train_num_all,1))        

    X_test = np.zeros((sum(Y>0)[0]-train_num_all,timestep,nb_features))   
    XP_test = np.zeros((sum(Y>0)[0]-train_num_all,w,w,num_PC)) 
    Y_test = np.zeros((sum(Y>0)[0]-train_num_all,1))     
    
    for i in range(1,n_class+1):
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        train_num = train_num_array[i-1]
        randomX = randomArray[i-1]        
        XP_train[flag1:flag1+train_num,:,:,:] = XP[index[randomX[0:train_num]],:,:,:]
        Y_train[flag1:flag1+train_num,0] = Y[index[randomX[0:train_num]],0]                       

        XP_test[flag2:flag2+n_data-train_num,:,:,:] = XP[index[randomX[train_num:n_data]],:,:,:]
        Y_test[flag2:flag2+n_data-train_num,0] = Y[index[randomX[train_num:n_data]],0]

        for j in range(0,timestep):
            X_train[flag1:flag1+train_num,j,:] = X[index[randomX[0:train_num]],j:j+(nb_features-1)*timestep+1:timestep]               
            X_test[flag2:flag2+n_data-train_num,j,:] = X[index[randomX[train_num:n_data]],j:j+(nb_features-1)*timestep+1:timestep]
        
        flag1 = flag1+train_num
        flag2 = flag2+n_data-train_num
                
    X_reshape = np.zeros((X.shape[0],timestep,nb_features))       
    for j in range(0,timestep):
        X_reshape[:,j,:] = X[:,j:j+(nb_features-1)*timestep+1:timestep]           
    X = X_reshape        
    return X.astype('float32'),X_train.astype('float32'),X_test.astype('float32'),XP.astype('float32'),XP_train.astype('float32'),XP_test.astype('float32'),Y.astype(int),Y_train.astype(int),Y_test.astype(int)
