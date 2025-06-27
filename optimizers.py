import numpy as np

def L_layer_model(X,y,layer_dims,learning_rate,iterations,print_cost=False):
    """
    Implement Gradient Descent 
    X->Input feature
    y->target
    layer_dims -> A list containg number of neurons in each layer 

    """
    parameters=initialize_parameters(layer_dims)
    costs=[]
    for i in range(iterations):
        AL,caches= L_layer_deep_forward_layer(X,parameters)
        cost=compute_cost(y,AL)
        grads=L_layer_backward(AL,y,caches)
        parameters=update_parameters(parameters,grads,learning_rate)
        costs.append(cost)
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return parameters,costs


def stochastic_gradient_descent(X,Y,layer_dims,learning_rate,epochs,print_cost=False):
     """
    Implement Stochastic Gradient Descent 
    X->Input feature
    y->target
    layer_dims -> A list containg number of neurons in each layer 

    """
    
    parameters=initialize_parameters(layer_dims)
    costs=[]
    m=X.shape[1]
    for i in range(epochs):
        cost=0
        for j in range(m):
            x=X[:,j].reshape(-1,1)
            y=Y[:,j].reshape(-1,1)
            AL,caches=L_layer_deep_forward_layer(x,parameters)
            cost+=compute_cost(y,AL)
            grads=L_layer_backward(AL,y,caches)
            parameters=update_parameters(parameters,grads,learning_rate)
        costs.append(cost/m)
        if print_cost:
            print(f"Cost after epoch {i}: {cost/m}")
    return parameters,costs

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    """
    Implement Gradient Descent with momentum
    X->Input feature
    y->target
    layer_dims -> A list containg number of neurons in each layer 
    parameters->A dict containg all parameters weights and bias
    grads->A dict containg gradient of all parameters 
    
    """
    L=len(parameters)//2
    for l in range(L):
        v["dW"+str(l+1)]=beta* v["dW"+str(l+1)] + (1-beta)* grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta* v["db"+str(l+1)] + (1-beta)* grads["db"+str(l+1)]
        
        parameters["W"+str(l+1)]-=learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)]-=learning_rate*v["db"+str(l+1)]
    return parameters,v
    
def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
     """
    Implement Adam Gradient Descent 
    X->Input feature
    y->target
    layer_dims -> A list containg number of neurons in each layer 
    parameters->A dict containg all parameters weights and bias
    grads->A dict containg gradient of all parameters 
    
    """
    v_corrected={}
    s_corrected={}
    
    L=len(parameters)//2
    
    for l in range(L):
        v["dW"+str(l+1)]=beta1*v["dW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]
        
        s["dW"+str(l+1)]=beta2*s["dW"+str(l+1)]+(1-beta2)*np.square(grads["dW"+str(l+1)])
        s["db"+str(l+1)]=beta2*s["db"+str(l+1)]+(1-beta2)*np.square(grads["db"+str(l+1)])

        v_corrected["dW"+str(l+1)]=v["dW"+str(l+1)]/(1-beta1**t)
        v_corrected["db"+str(l+1)]=v["db"+str(l+1)]/(1-beta1**t)
        s_corrected["dW"+str(l+1)]=s["dW"+str(l+1)]/(1-beta2**t)
        s_corrected["db"+str(l+1)]=s["db"+str(l+1)]/(1-beta2**t)

        parameters["W"+str(l+1)]-=learning_rate*v_corrected["dW"+str(l+1)]/np.sqrt(s_corrected["dW"+str(l+1)]+epsilon)
        parameters["b"+str(l+1)]-=learning_rate*v_corrected["db"+str(l+1)]/np.sqrt(s_corrected["db"+str(l+1)]+epsilon)
    return parameters,v,s

          
def accuracy_of_model(X,y,parameters):
    """ 
    Calculate Accuracy of prediction of neural Network
    X->Input feature
    y->target
    parameters->A dict containg all parameters weights and bias
    
    """
    m=y.shape[1]
    y_pred,cache= L_layer_deep_forward_layer(X,parameters)
    y_pred=(y_pred[0,:]>0.5).astype(int).reshape(1,-1)
    count=np.sum(y_pred[0,:]==y[0,:])
    accuracy=count/m

    return (accuracy)

