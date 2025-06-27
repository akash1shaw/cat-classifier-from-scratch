import numpy as np

def sigmoid(Z):
    """
    Implements sigmoid non linear activation to linear output
    Z-> linear output
    """
    A=1/(1+np.exp(-Z))
    cache=Z
    return A,cache

def relu(Z):
    """
    Implements relu non linear activation to linear output
    Z-> linear output
    """
    A=np.maximum(0,Z)
    cache=Z
    return A,cache

def tanh(Z):
    """
    Implements tanh non linear activation to linear output
    Z-> linear output
    """
    A=np.tanh(Z)
    cache=Z
    return A,cache

def linear_activation_layer(A_prev,W,b,activation):

    """
    linear_activation_layer has four parameters X,W,b,activatuion
    A_prev-> input feature matrix
    W-> weight matirx
    b-> bias matrix
    activation-> applied non-linearity
    
    """
    Z=np.matmul(W,A_prev)+b
    linear_cache=(A_prev,W,b)
    
    if activation=="relu":
        A,activation_cache=relu(Z)
        
    elif activation=="sigmoid":
        A,activation_cache=sigmoid(Z)
        
    elif activation=="tanh":
        A,activation_cache=tanh(Z)
        
    cache=(linear_cache,activation_cache)
    assert(Z.shape==A.shape)
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    return A,cache    

def L_layer_deep_forward_layer(X,parameters):
    """ 
    L_layer_deep_forward_layer implements forward propagation part of Neural Netwrok
    X -> input feature matrix
    parameters -> A set containing all weight matrix and bias vector
    
    """

    
    L=len(parameters)//2
    A_in=X
    caches=[]
    
    for i in range(1,L):
        A_out,cache=linear_activation_layer(A_in,parameters["W"+str(i)],parameters["b"+str(i)],"relu")
        A_in=A_out
        caches.append(cache)
        
    A_L,cache=linear_activation_layer(A_in,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    
    return A_L,caches

def compute_cost(y,y_pred):
    """
    Implement the binary crossntropy function 
    y-> target label
    y_pred-> predicted label
    
    """
    loss=-(y*np.log(y_pred+1e-15)+(1-y)*np.log(1-y_pred+1e-15))
    
    cost=np.mean(loss)
    cost=np.squeeze(cost)
    return cost