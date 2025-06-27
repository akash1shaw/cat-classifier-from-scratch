import numpy as np
def linear_activation_layer_dropout(A_prev,W,b,activation,keep_prop):
    Z=np.matmul(W,A_prev)+b
    linear_cache=(A_prev,W,b)
    if activation=="relu":
        A,activation_cache=relu(Z)
        
    elif activation=="sigmoid":
        A,activation_cache=sigmoid(Z)
        
    elif activation=="tanh":
        A,activation_cache=tanh(Z)
    D=(np.random.rand(A.shape[0],A.shape[1])<keep_prop).astype(int)

    A=A*D
    A/=keep_prop
 
    
    cache=(linear_cache,activation_cache,D)
    return A,cache

def L_layer_deep_forward_layer_dropout(X,parameters,keep_prop):
    L=len(parameters)//2
    A_prev=X
    caches=[]
    for i in range(L-1):
        A,cache=linear_activation_layer_dropout(A_prev,parameters["W"+str(i+1)],parameters["b"+str(i+1)],"relu",keep_prop)
        caches.append(cache)
        A_prev=A
        
    A,cache=linear_activation_layer_dropout(A_prev,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid",1)
    caches.append(cache)
    return A,caches

def linear_activation_backward_dropout(dA,cache,activation,keep_prop):
    linear_cache,activation_cache,D=cache
    dA=dA*D/keep_prop
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        
        
    elif activation=="tanh":
        dZ=tanh_backward(dA,activation_cache)
    
    dW,db,dA_prev=linear_backward(dZ,linear_cache)  
    return dW,db,dA_prev  


def L_layer_backward_dropout(A_L,y,caches,keep_prop):
    grads={}
    L=len(caches)
    cache=caches[L-1]
    m=y.shape[1]
    
    dAL=(-np.divide(y,A_L+1e-8)+np.divide(1-y,1-A_L+1e-8))/m
    grads["dW"+str(L)],grads["db"+str(L)],grads["dA"+str(L-1)]=linear_activation_backward_dropout(dAL,cache,"sigmoid",1)
 
    
    for i in reversed(range(L-1)):
        
        cache=caches[i]
     
        grads["dW"+str(i+1)],grads["db"+str(i+1)],grads["dA"+str(i)]=linear_activation_backward_dropout(grads["dA"+str(i+1)],cache,"relu",keep_prop)
        
    
    return grads

def L_layer_model_dropout(X,y,layer_dims,learning_rate,iterations,keep_prop,print_cost=False):
    parameters=initialize_parameters(layer_dims)
    costs=[]
    for i in range(iterations):
        AL,caches= L_layer_deep_forward_layer_dropout(X,parameters,keep_prop)
        cost=compute_cost(y,AL)
        grads=L_layer_backward_dropout(AL,y,caches,keep_prop)
        parameters=update_parameters(parameters,grads,learning_rate)
        costs.append(cost)
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
            if(i>0 and abs(costs[i]-costs[i-1])<1e-5):
                break
    return parameters,costs