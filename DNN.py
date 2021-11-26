# Deep Neural Network Design (for adversarial training), with Self Scaling Formulation:
# 
# Reid Zaffino (zaffino@uwindsor.ca) 2021-09
# 
# Resources from Coursera "Neural Networks and Deep Learning" Course
# 
# Self Scaling Methodology from: Djahanshahi, Hormoz., "A robust hybrid VLSI neural network architecture for a smart optical sensor." (1999). Electronic Theses and Dissertations. 737. https://scholar.uwindsor.ca/etd/737

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import cPickle 

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def sigmoid(Z):
    
    return np.divide(1, 1 + np.exp(-Z)), Z

def tanh(Z):
    
    #return Z * (Z > 0), Z
    return np.tanh(Z), Z

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W,A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, ss):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "tanh"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        
        if(ss):
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z/len(A_prev))
        
        else:
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
    
    elif activation == "tanh":

        if(ss):
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z/len(A_prev))
        
        else:
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, ss):
    """
    Implement forward propagation for the [LINEAR->tanh]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> tanh]*(L-1). Add "cache" to the "caches" list.
    # The for loop starts at 1 because layer 0 is the input
    for l in range(1, L):
        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "tanh", ss)
        caches.append(cache)
        
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid", ss)
    caches.append(cache)
          
    return AL, caches

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (-1/m)*np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def sigmoid_backward(dA, activation_cache):
    
    return np.multiply(dA, np.multiply(np.divide(1, 1 + np.exp(-activation_cache)), 1 - np.divide(1, 1 + np.exp(-activation_cache))))

def tanh_backward(dA, activation_cache):
    
    #return np.multiply(dA, (activation_cache > 0) * 1)
    return np.multiply(dA, 1 - np.tanh(activation_cache)**2)

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "tanh"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "tanh":
        
        dZ =  tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->tanh] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "tanh" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Initializing the backpropagation
    
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (tanh -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "tanh")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = params.copy()
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        
    return parameters

def predict(X, Y, parameters, ss):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (0 / 1)
    """
    
    A, cache = L_model_forward(X, parameters, ss)
    
    incorrect = (A > 0.5).astype(int) == Y
    sum = 0
    
    for i in range (incorrect.shape[1]):
        sum = sum + incorrect[0][i]
    
    return sum/Y.shape[1]

def L_layer_model(X, Y, Xt, Yt, layers_dims, wf, learning_rate = 1, num_iterations = 3000, print_cost=False, ss=False):
    """
    Implements a L-layer neural network: [LINEAR->tanh]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    start = time.time()

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> tanh]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, ss)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            curr = time.time()
            print(str(i) + "\t\t\t" +  str(np.squeeze(cost)) + "\t\t" + str(predict(X, Y, parameters, ss)) + "\t\t" + str(predict(Xt, Yt, parameters, ss)) + "\t\t\t" + str(curr - start))
            wf.write(str(i) + "\t\t\t" +  str(np.squeeze(cost)) + "\t\t" + str(predict(X, Y, parameters, ss)) + "\t\t" + str(predict(Xt, Yt, parameters, ss)) + "\t\t\t" + str(curr - start) + "\n")
            
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs

def run(learning_rate, layers_dims, num_iterations, ss, wf):
    np.random.seed(1)

    dir_path =  os.path.realpath('./')
    file1 = dir_path + "\cifar-10-batches-py\data_batch_1"
    file2 = dir_path + "\cifar-10-batches-py\data_batch_2"
    file3 = dir_path + "\cifar-10-batches-py\data_batch_3"
    file4 = dir_path + "\cifar-10-batches-py\data_batch_4"
    file5 = dir_path + "\cifar-10-batches-py\data_batch_5"
    filet = dir_path + "\cifar-10-batches-py\\test_batch"

    f1 = open(file1, 'rb')
    data1 = cPickle.load(f1, encoding = 'latin1')
    f1.close()
    X1 = data1["data"] 
    Y1 = data1["labels"]

    f2 = open(file2, 'rb')
    data2 = cPickle.load(f2, encoding = 'latin1')
    f2.close()
    X2 = data2["data"]
    Y2 = data2["labels"]

    f3 = open(file3, 'rb')
    data3 = cPickle.load(f3, encoding = 'latin1')
    f3.close()
    X3 = data3["data"]
    Y3 = data3["labels"]

    f4 = open(file4, 'rb')
    data4 = cPickle.load(f4, encoding = 'latin1')
    f4.close()
    X4 = data4["data"]
    Y4 = data4["labels"]

    f5 = open(file5, 'rb')
    data5 = cPickle.load(f5, encoding = 'latin1')
    f5.close()
    X5 = data2["data"]
    Y5 = data2["labels"]

    ft = open(filet, 'rb')
    datat = cPickle.load(ft, encoding = 'latin1')
    ft.close()
    Xt = datat["data"]
    Yt = datat["labels"]

    nonbin1 = []
    for i in range (len(Y1)):
        if (Y1[i] != 0 and Y1[i] != 1):
            nonbin1.append(i)

    nonbin2 = []
    for i in range (len(Y2)):
        if (Y2[i] != 0 and Y2[i] != 1):
            nonbin2.append(i)

    nonbin3 = []
    for i in range (len(Y3)):
        if (Y3[i] != 0 and Y3[i] != 1):
            nonbin3.append(i)

    nonbin4 = []
    for i in range (len(Y4)):
        if (Y4[i] != 0 and Y4[i] != 1):
            nonbin4.append(i)

    nonbin5 = []
    for i in range (len(Y5)):
        if (Y5[i] != 0 and Y5[i] != 1):
            nonbin5.append(i)

    nonbint = []
    for i in range (len(Yt)):
        if (Yt[i] != 0 and Yt[i] != 1):
            nonbint.append(i)

    X1 = np.delete(X1, nonbin1, axis = 0)
    Y1 = np.delete(Y1, nonbin1)

    X2 = np.delete(X2, nonbin2, axis = 0)
    Y2 = np.delete(Y2, nonbin2)

    X3 = np.delete(X3, nonbin3, axis = 0)
    Y3 = np.delete(Y3, nonbin3)

    X4 = np.delete(X4, nonbin4, axis = 0)
    Y4 = np.delete(Y4, nonbin4)

    X5 = np.delete(X5, nonbin5, axis = 0)
    Y5 = np.delete(Y5, nonbin5)

    Xt = np.delete(Xt, nonbint, axis = 0)
    Yt = np.delete(Yt, nonbint)

    X1 = (X1.reshape(X1.shape[0], -1)/255.).T
    Y1 = Y1.reshape(1, Y1.shape[0])

    X2 = (X2.reshape(X2.shape[0], -1)/255.).T
    Y2 = Y2.reshape(1, Y2.shape[0])

    X3 = (X3.reshape(X3.shape[0], -1)/255.).T
    Y3 = Y3.reshape(1, Y3.shape[0])

    X4 = (X4.reshape(X4.shape[0], -1)/255.).T
    Y4 = Y4.reshape(1, Y4.shape[0])

    X5 = (X5.reshape(X5.shape[0], -1)/255.).T
    Y5 = Y5.reshape(1, Y5.shape[0])

    Xt = (Xt.reshape(Xt.shape[0], -1)/255.).T
    Yt = Yt.reshape(1, Yt.shape[0])

    X1 = np.append(X1, X2, axis = 1)
    X3 = np.append(X3, X4, axis = 1)
    X = np.append(X1, X3, axis = 1)
    X = np.append(X, X5, axis = 1)

    Y1 = np.append(Y1, Y2, axis = 1)
    Y3 = np.append(Y3, Y4, axis = 1)
    Y = np.append(Y1, Y3, axis = 1)
    Y = np.append(Y, Y5, axis = 1)

    parameters, costs = L_layer_model(X, Y, Xt, Yt, layers_dims, wf, learning_rate, num_iterations, print_cost = True, ss = ss)

def main():

    num_iterations = 10000
    ss = True
    base_rate = 0.1
    learning_rate = base_rate
    hl = 1
    layers_dims = [3072, hl, 1]
    dir_path =  os.path.realpath('./')
    
    for i in range(10):
        for j in range(5):
            learning_rate = base_rate * (1 + j)
            hl = 2**(i)
            layers_dims = [3072, hl, 1]
            wfile = dir_path + "\cifar-10-batches-py\out_ss_" + str(ss) + "_lr_" + str(learning_rate) + "_hl_" + str(hl) + ".txt"
            wf = open(wfile, "w")
            print("out_ss_" + str(ss) + "_lr_" + str(learning_rate) + "_hl_" + str(hl) + ".txt")
            wf.write("out_ss_" + str(ss) + "_lr_" + str(learning_rate) + "_hl_" + str(hl) + "\n")
            print("Iteration\t\tCost\t\t\t\tTraining Accuracy\t\tTest Accuracy\t\tTime")
            wf.write("Iteration\t\tCost\t\t\t\tTraining Accuracy\t\tTest Accuracy\t\tTime\n")
            run(learning_rate , layers_dims, num_iterations, ss, wf)
            wf.close()
    


if __name__ == "__main__":
    main()