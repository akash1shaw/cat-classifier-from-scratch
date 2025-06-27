def binaryCrossentropyRegularized(y,AL,lambd,parameters):
    m=y.shape[1]
    regularized_term=0
    L=len(parameters)//2
    for i in range(L):
        regularized_term+=np.sum(parameters["W"+str(i+1)]**2)
    regularized_term*=lambd/(2*m)
   
    cost_term=-y*np.log(AL+1e-8)-(1-y)*np.log(1-AL+1e-8)
    cost_term=np.sum(cost_term)
    return cost_term/m+regularized_term




def linear_backward_regularized(dZ,linear_cache,lambd):
    m=dZ.shape[1]
    A_prev,W,b=linear_cache

    dW=np.matmul(dZ,A_prev.T)/m+lambd*W/m
    db=np.mean(dZ,axis=1,keepdims=True)
    dA_prev=np.matmul(W.T,dZ)
    return dW,db,dA_prev
    

def linear_activation_backward_regularized(dA,cache,activation,lambd):
    linear_cache,activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        
        
    elif activation=="tanh":
        dZ=tanh_backward(dA,activation_cache)
    dW,db,dA_prev=linear_backward_regularized(dZ,linear_cache,lambd)
    return dW,db,dA_prev

def L_layer_backward_regularized(A_L,y,caches,lambd):
    m=y.shape[1]
    L=len(caches)
    cache=caches[L-1]
    grads={}
    
    dA=-y/(A_L+1e-8)+(1-y)/(1-A_L+1e-8)
    grads["dW"+str(L)],grads["db"+str(L)],grads["dA"+str(L-1)]=linear_activation_backward_regularized(dA,cache,"sigmoid",lambd)
    for i in reversed(range(1,L)):
        cache=caches[i-1]
        grads["dW"+str(i)],grads["db"+str(i)],grads["dA"+str(i-1)]=linear_activation_backward_regularized(grads["dA"+str(i)],cache,"relu",lambd)
    return grads

def update_parameters_regularized(parameters,grads,learning_rate,lambd):
    L=len(parameters)//2
    for i in range(L):
        parameters["W"+str(i+1)]-=learning_rate*grads["dW"+str(i+1)]
        parameters["b"+str(i+1)]-=learning_rate*grads["db"+str(i+1)]
    return parameters

def L_layer_model_regularized(X,y,layer_dims,learning_rate,iteration,lambd,print_cost=False):
    parameters=initialize_parameters(layer_dims)
    costs=[]
    for i in range(iteration):
        AL,caches= L_layer_deep_forward_layer(X,parameters)
        cost=binaryCrossentropyRegularized(y,AL,lambd,parameters)
        grads=L_layer_backward_regularized(AL,y,caches,lambd)
        parameters=update_parameters_regularized(parameters,grads,learning_rate,lambd)
        if print_cost and i%100==0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")
    return parameters,costs