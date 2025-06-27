import numpy as np


def linear_backward(dZ,linear_cache):
    """
    Implement 'single layer' linear backpropagation 
    dZ-> gradient of cost function w.r.t to linear output
    cache-> it is a tuple coming from linear activation layer 
    
    """
    m=dZ.shape[1]
    A_prev,W,b=linear_cache
    
    dW=np.matmul(dZ,A_prev.T)/m
    db=np.mean(dZ,axis=1,keepdims=True)
    dA_prev=np.matmul(W.T,dZ)
    
    return dW,db,dA_prev

def sigmoid_backward(dA,activation_cache):
    """
    Implement backward function for sigmoid 
    dA-> gradient of cost function with respect to activation output
    activation_cache->It is a tuple coming from activation layer and part of cache

    """
    Z=activation_cache    
    A=1/(1+np.exp(-Z))
     return dA*A*(1-A)

def relu_backward(dA,activation_cache):
    """
    Implement backward function for relu
    dA-> gradient of cost function with respect to activation output
    activation_cache->It is a tuple coming from activation layer and part of cache

    """
    Z=activation_cache
    return dA*np.where(Z>0,1,0)

def tanh_backward(dA,activation_cache):
        """
    Implement backward function for sigmoid 
    dA-> gradient of cost function with respect to activation output
    activation_cache->It is a tuple coming from activation layer and part of cache

    """
    Z=activation_cache
    return dA*(1-np.tanh(Z)**2)

def linear_activation_backward(dA,cache,activation):
    
    linear_cache,activation_cache=cache
    
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        
        
    elif activation=="tanh":
        dZ=tanh_backward(dA,activation_cache)
    
    dW,db,dA_prev=linear_backward(dZ,linear_cache)  
    return dW,db,dA_prev

print(caches)

def L_layer_backward(A_L,y,caches):
    """
    Implement backpropagation for L layers of neural netwrok 
    A_L->output prediction 
    y-> target 
    caches-> A list containing all cache for all parameters required for backpropagation
    """
    grads={}
    L=len(caches)
    cache=caches[L-1]
    m=y.shape[0]
    
    dAL=(-np.divide(y,A_L+1e-15)+np.divide(1-y,1-A_L+1e-15))
    grads["dW"+str(L)],grads["db"+str(L)],grads["dA"+str(L-1)]=linear_activation_backward(dAL,cache,"sigmoid")
 
    
    for i in reversed(range(L-1)):

        
        cache=caches[i]
     
        grads["dW"+str(i+1)],grads["db"+str(i+1)],grads["dA"+str(i)]=linear_activation_backward(grads["dA"+str(i+1)],cache,"relu")
        
    
    return grads
def update_parameters(parameters,grads,learning_rate):
    """
    Implement updation of paramters during one itetation of gradient descent 
    parameters->A dict containg all parameters weights and bias
    grads->A dict containg gradient of all parameters 
    learning rate 
    """
    L=len(parameters)//2
    for i in range (L):
        parameters["W"+str(i+1)]-=learning_rate*grads["dW"+str(i+1)]
        parameters["b"+str(i+1)]-=learning_rate*grads["db"+str(i+1)]
    return parameters


def model(X,y,layer_dims,optimizer,learning_rate=0.0007,mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,epsilon=1e-8,num_epochs=10000,print_cost=True):

    parameters=initialize_parameters(layer_dims)
    L=len(parameters)//2
    costs=[]
    t=0
    seed=1
    m=y.shape[1]
    
    print(f"The number of training example is: {m}")
    print(f"The size of mini batch for each epoch is: {mini_batch_size}")
    
    
    if optimizer=="GD":
        pass
    elif optimizer=="momentum":
        v=initialize_velocity(parameters)
    elif optimizer=="Adam":
        v,s=initialize_Adam(parameters)
    
    for i in range(num_epochs):
        seed=seed+1
        mini_batches=random_mini_batches(X,y,mini_batch_size,seed)
        total_cost=0

        for mini_batch in mini_batches:
            random_X,random_y=mini_batch
            
            AL,caches=L_layer_deep_forward_layer(random_X,parameters)
            total_cost+=compute_cost(random_y,AL)
            grads=L_layer_backward(AL,random_y,caches)

            if optimizer=="GD":
                parameters=update_parameters(parameters,grads,learning_rate)
                
            elif optimizer=="momentum":
                parameters,v=update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
                
            elif optimizer=="Adam":
                t+=1
                parameters,v,s=update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8)
        cost_avg=total_cost/m
            
        if i%100==0:
            costs.append(cost_avg)
        if i%100==0 and print_cost:
            print(f"cost after epoch {i}: {cost_avg}")
                
    plt.plot(costs)
    plt.xlabel("# epoch (per 100)")
    plt.ylabel("avg_cost")
    plt.title("learning rate" +str(learning_rate))
    plt.show()
    return parameters