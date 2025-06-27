import numpy as np

def initialize_parameters(layer_dims):
    """ 
    "layer dims" is a list consiting of number of input features and hidden units in each layer
    initalize_parameters() initializes parameters of all hidden layers 

    >>> initalize_parameters([4,10,15,25,10])
            return parameters
    """
    
    np.random.seed(42)
    L=len(layer_dims)
    parameters={}

    
    for i in range(1,L):
        
        parameters["W"+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)]=np.zeros((layer_dims[i],1))
        
        assert(parameters["W"+str(i)]).shape==(layer_dims[i],layer_dims[i-1])
        assert(parameters["b"+str(i)]).shape==(layer_dims[i],1)
        
    return parameters
    
def initialize_velocity(parameters):
    """
    Initialize v matrix for calculation moving averages
    parameters->A dict containg all parameters weights and bias
    
    """
    L=len(parameters)//2
    v={}
    for i in range(L):
        v["dW"+str(i+1)]=np.zeros((parameters["W"+str(i+1)].shape[0],parameters["W"+str(i+1)].shape[1]))
        v["db"+str(i+1)]=np.zeros((parameters["b"+str(i+1)].shape[0],parameters["b"+str(i+1)].shape[1]))
    return v
def initialize_Adam(parameters):
     """
    Initialize v and s matrix for calculation moving averages
    parameters->A dict containg all parameters weights and bias
    
    """
    L=len(parameters)//2
    v={}
    s={}
    for l in range(L):
        v["dW"+str(l+1)]=np.zeros((parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
        v["db"+str(l+1)]=np.zeros((parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))

        s["dW"+str(l+1)]=np.zeros((parameters["W"+str(l+1)].shape[0], parameters["W"+str(l+1)].shape[1]))
        s["db"+str(l+1)]=np.zeros((parameters["b"+str(l+1)].shape[0], parameters["b"+str(l+1)].shape[1]))
    return v,s
    