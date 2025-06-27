import numpy as np
def random_mini_batches(X,y,mini_batch_size,seed):
    """ 
    X->Input feature
    y->target 
    mini_batch_size -> size of ezh mini batch
    seed->to ensure random shuffling of batches in each epochs 
    
    """
    np.random.seed(seed)
    mini_batches=[]
    m=X.shape[1]
    
    permutation=list(np.random.permutation(m))
    random_X=X[:,permutation]
    random_y=y[:,permutation]
    
    num_complete_batch=(m//mini_batch_size)
    
    for k in range(num_complete_batch):
        mini_batch_X=random_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_y=random_y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        
        batch=(mini_batch_X,mini_batch_y)
        mini_batches.append(batch)
        
    if(m%mini_batch_size!=0):
        mini_batch_X=random_X[:,num_complete_batch*mini_batch_size:]
        mini_batch_y=random_y[:,num_complete_batch*mini_batch_size:]
        batch=(mini_batch_X,mini_batch_y)
        mini_batches.append(batch)
        
    return mini_batches