# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:44:10 2018

@author: amcle
"""

import numpy as np
import tensorflow as tf

# Importing model to load .mat files of MATLAB formatted data
from scipy.io import loadmat


# Implements the weight mapping function of weight_mapping.m
# Input:    theta := Unconstrained weights matrix, 
#           K := Constant number that is the difference between σ_max and σ_min
# Output:   Constrained mapped weight matrix
# MATLAB source file:   weight_mapping.m
def weight_mapping(theta, K):
    theta = 1.0 + np.divide(K, 1.0 + np.exp(-theta))
    sum_theta = np.sum(theta, axis=1)
    
    # np.tile repeats the row 394 times then transposes to get the right dimension
    return np.divide(theta, np.tile(sum_theta, (theta.shape[1], 1)).transpose())


# Calculates the sigmoid activation function at z, combines sigmoid.m, sigmoidn.m, and sigmoidOut.m functions
# as they are very similar to each other and can be combined by adding an extra argument
# Input:    z := Net inputs into neurons in a specific layer, 
#           cc := List of activation parameters taken from HSpice, 
#           Vdd := Constant, voltage used in neuromorphic circuit, 
#           nmos := Boolean to determine whether to implement sigmoidn.m or sigmoid.m functionality
# Output:   Activated neuron output
# MATLAB source file:   sigmoid.m, sigmoidn.m, sigmoidOut.m
def sigmoid(z, cc, Vdd, nmos):
    sharp_factor = cc[3]
    
    if nmos:
        half_Vdd = Vdd / 2
    else:
        half_Vdd = -(Vdd / 2)
    
    return np.multiply(half_Vdd, np.tanh(np.multiply(sharp_factor, np.subtract(z, cc[2]))))


# Calculates the sigmoid gradient activation function at z, combines sigmoidGradient.m and sigmoidGradientn.m 
# functions as they are very similar to each other and can be combined by adding an extra argument
# Input:    z := Net inputs into the neurons in a specific layer, 
#           cc := List of activation parameters taken from HSpice, 
#           nmos := Boolean to determine whether to implement sigmoidGradientn.m or sigmoidGradient.m
#           functionality
# Output:   Activated neuron output
# MATLAB source file:   sigmoidGradient.m, sigmoidGradientn.m
def sigmoidGradient(z, cc, nmos):
    if nmos:
        return np.multiply(cc[3], np.subtract(1, np.square(np.tanh(np.multiply(cc[3], np.subtract(z, cc[2]))))))
    else:
        return np.multiply(-cc[3], np.subtract(1, np.square(np.tanh(np.multiply(cc[3], np.subtract(z, cc[2]))))))


# Randomly initializes the weight of a layer with L_in incoming connections and L_out outgoing connections, 
# allowing the symmetry of the layer to be broken while training the neural network
# Input:    L_in := Incoming connections of a layer, 
#           L_out := Outgoing connections of a layer
# Output:   Returns random initial weights of small values with size(L_out, L_in + 1) and the columns handling  
#           the "bias" terms and the first row corresponding to the parameters for the bias units
# MATLAB source file:   randInitializeWeights.m
def randInitializeWeights(L_in, L_out):
    # W = np.zeros([L_out, 1 + L_in])
    epsilon = np.sqrt(6 / (L_in + L_out))
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon


# Implements physical circuit constraints on the weight mapping that occurs during backpropagation of the network
# Input:    theta := Unconstrained weights matrix, 
#           K := Constant number that is the difference between σ_max and σ_min
# Output:   Activated neuron output
# MATLAB source file:   weight_md.m
def weight_md(theta, K):
    num_pn = theta.shape[0]
    num_n = theta.shape[1]
    g1 = np.multiply(K, np.multiply(np.divide(1.0, 1.0 + np.exp(-theta)), 1.0 - np.divide(1.0, 1.0 + np.exp(-theta))))
    g1 = np.kron(g1, np.ones((num_n, 1))).transpose()
    
    theta = 1.0 + np.divide(K, 1.0 + np.exp(-theta))
    sum_theta = np.sum(theta, axis=1).reshape(-1, 1)
    sums_theta1 = np.tile(sum_theta, (1, theta.shape[1]))
    theta = np.kron(theta, np.ones((num_n, 1))).transpose()
    sums_theta = np.kron(sums_theta1, np.ones((num_n, 1))).transpose()
    sums_theta1 = sums_theta1.transpose()
    sums_theta1 = np.divide(1.0, sums_theta1.flatten('F').reshape(-1, 1))
    
    # Probably not the best solution but it works
    diag1 = np.arange(0, num_n * num_n, num_n + 1)
    ind = diag1.tolist()
    for i in range(1, num_pn):
        ind.extend(np.add(num_n * num_n * i, diag1).tolist())
    diag_nums = np.array(ind)
    g = np.divide(-theta, np.square(sums_theta))
    g[np.unravel_index(diag_nums, g.shape, 'F')] = g[np.unravel_index(diag_nums, g.shape, 'F')] + sums_theta1.transpose()
    # print(num_pn, num_n, g1.shape, sum_theta.shape, sums_theta1.shape, diag_nums.shape, g.shape)
    
    return g


# 
# For weights constrained, K[0] for Theta1_mapping, K[1] for Theta_2_1 mapping if existed, K[2] for Theta_2_2 
# mapping if existed, and K[3] for Theta_2 mapping
# Input:    
# Output:   
# MATLAB source file:   my_backpropagation.m
def my_backpropagation(cc, cc2, lbda, J, sharp_factor, X, Y, input_layer_size, hidden_layer_size, num_labels, K, Vdd):
    X = X.transpose()       # Line 57: my_backpropagation.m
    Y = Y.transpose()       # Line 58: my_backpropagation.m
    # print(X.shape)
    m = X.shape[0]          # Line 59: my_backpropagation.m
    
    # For weights constrained
    Ron = 100               # Line 61: my_backpropagation.m
    Roff = 16e3             # Line 62: my_backpropagation.m
    rat = Roff / Ron        # Line 63: my_backpropagation.m
    # print(X.shape, Y.shape)
    
    # Temporary values for theta to make sure weight_mapping function is working properly
    # Theta1 = loadmat("theta.mat")["Theta1"]
    # Theta1n = loadmat("theta.mat")["Theta1n"]
    # Theta2 = loadmat("theta.mat")["Theta2"]
    # Theta2n = loadmat("theta.mat")["Theta2n"]
    # print(Theta1.shape, Theta1n.shape, Theta2.shape, Theta2n.shape)
    
    # Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)         # Line 66: my_backpropagation.m
    # Theta1n = randInitialzieWeights(input_layer_size, hidden_layer_size)        # Line 67: my_backpropagation.m
    # Theta2 = randInitializeWeights(hidden_layer_size, num_labels)               # Line 88: my_backpropagation.m
    # Theta2n = randInitializeWeights(hidden_layer_size, num_labels)              # Line 89: my_backpropagation.m
    
    # Defining the unconstrained weights in a dictionary
    nn_params = {
            "Theta1": randInitializeWeights(input_layer_size, hidden_layer_size),       # Line 66: my_backpropagation.m
            "Theta1n": randInitializeWeights(input_layer_size, hidden_layer_size),      # Line 67: my_backpropagation.m
            "Theta2": randInitializeWeights(hidden_layer_size, num_labels),             # Line 88: my_backpropagation.m
            "Theta2n": randInitializeWeights(hidden_layer_size, num_labels),            # Line 89: my_backpropagation.m
            "Theta1_grad": np.zeros((hidden_layer_size, input_layer_size + 1)),         # Line 78: nnCostFunction2.m
            "Theta2_grad": np.zeros((num_labels, hidden_layer_size + 1)),               # Line 79: nnCostFunction2.m
            "Theta1n_grad": np.zeros((hidden_layer_size, input_layer_size + 1)),        # Line 80: nnCostFunction2.m
            "Theta2n_grad": np.zeros((num_labels, hidden_layer_size + 1)),              # Line 81: nnCostFunction2.m
            "a_1": np.zeros((1, input_layer_size + 1)),                                 # Line 104: nnCostFunction2.m
            "z_2": np.zeros((1, hidden_layer_size + 1)),                                # Line 105: nnCostFunction2.m
            "a_2": np.zeros((1, hidden_layer_size + 1)),                                # Line 106: nnCostFunction2.m
            "z_3": np.zeros((1, num_labels)),                                           # Line 107: nnCostFunction2.m
            "a_3": np.zeros((1, num_labels)),                                           # Line 108: nnCostFunction2.m
            "delta_3": np.zeros((num_labels, 1)),                                       # Line 109: nnCostFunction2.m
            "delta_2": np.zeros((hidden_layer_size, 1)),                                # Line 110: nnCostFunction2.m
            "Delta_2": np.zeros((num_labels, hidden_layer_size + 1)),                   # Line 111: nnCostFunction2.m
            "Delta_1": np.zeros((hidden_layer_size, input_layer_size + 1))              # Line 112: nnCostFunction2.m
            }
    # print(Theta1_grad.shape, Theta1n_grad.shape, Theta2_grad.shape, Theta2n_grad.shape)
    # print(a_1.shape, z_2.shape, a_2.shape, z_3.shape, a_3.shape, delta_3.shape, delta_2.shape, Delta_2.shape, Delta_1.shape)
    
    # nn_params = {
    #         "Theta1": randInitializeWeights(input_layer_size, hidden_layer_size),       # Line 66: my_backpropagation.m
    #         "Theta1n": randInitializeWeights(input_layer_size, hidden_layer_size),      # Line 67: my_backpropagation.m
    #         "Theta2": randInitializeWeights(hidden_layer_size, num_labels),             # Line 88: my_backpropagation.m
    #         "Theta2n": randInitializeWeights(hidden_layer_size, num_labels),            # Line 89: my_backpropagation.m
    #         "Theta1_grad": tf.Variable(tf.zeros([hidden_layer_size, input_layer_size + 1])), 
    #         "Theta1n_grad": tf.Variable(tf.zeros([hidden_layer_size, input_layer_size + 1])), 
    #         "Theta2_grad": tf.Variable(tf.zeros([num_labels, hidden_layer_size + 1])), 
    #         "Theta2n_grad": tf.Variable(tf.zeros([num_labels, hidden_layer_size + 1])), 
    #         "a_1": tf.Variable(tf.zeros([1, input_layer_size + 1])), 
    #         "z_2": tf.Variable(tf.zeros([1, hidden_layer_size + 1])), 
    #         "a_2": tf.Variable(tf.zeros([1, hidden_layer_size + 1])), 
    #         "z_3": tf.Variable(tf.zeros([1, num_labels])), 
    #         "a_3": tf.Variable(tf.zeros([1, num_labels])), 
    #         "delta_3": tf.Variable(tf.zeros([num_labels, 1])), 
    #         "delta_2": tf.Variable(tf.zeros([hidden_layer_size, 1])), 
    #         "Delta_1": tf.Variable(tf.zeros([hidden_layer_size, input_layer_size + 1])), 
    #         "Delta_2": tf.Variable(tf.zeros([num_labels, hidden_layer_size + 1]))
    #         }
    # for key in nn_params:
    #     print(key, nn_params[key].get_shape().as_list())\
    
    nnCostFunction(cc, cc2, lbda, J, sharp_factor, X, Y, K, Vdd, m, nn_params)


# 
# Input:    J := ,
#           nn_params := Dictionary ,
#           X := ,
#           Y := 
# Output:   
# MATLAB source file:   nnCostFunction2.m
def nnCostFunction(cc, cc2, lbda, J, sharp_factor, X, Y, K, Vdd, m, nn_params):
    # Defining the positive and negative bias terms respectively
    pbias = np.full((m, 1), Vdd / 2)
    nbias = np.full((m, 1), -Vdd / 2)
    
    X1 = np.concatenate((pbias, X), axis=1)     # Line 145: nnCostFunction2.m
    X2 = np.concatenate((nbias, X), axis=1)     # Line 146: nnCostFunction2.m
    # print(X1.shape, X2.shape, nbias.shape, pbias.shape)
    
    weights1 = weight_mapping(np.concatenate((nn_params["Theta1"], nn_params["Theta1n"]), axis=1), K[0])        # Line 148: nnCostFunction2.m
    # Line 150: nnCostFunction2.m
    sigma1 = np.matmul(X1, weights1[:nn_params["Theta1"].shape[0], :nn_params["Theta1"].shape[1]].transpose()) + np.matmul(X2, weights1[:nn_params["Theta1"].shape[0], nn_params["Theta1"].shape[1]:].transpose())
    h1 = sigmoid(sigma1, cc, Vdd, False)            # Line 152: nnCostFunction2.m
    # print(weights1.shape, sigma1.shape, h1.shape)
    
    weights2 = weight_mapping(np.concatenate((nn_params["Theta2"], nn_params["Theta2n"]), axis=1), K[3])        # Line 181: nnCostFunction2.m
    # Line 183: nnCostFunction2.m
    sigmaO = np.matmul(np.concatenate((pbias, h1), 1), weights2[:nn_params["Theta2"].shape[0], :nn_params["Theta2"].shape[1]].transpose()) + np.matmul(np.concatenate((nbias, sigmoid(sigma1, cc2, Vdd, True)), 1), weights2[:nn_params["Theta2"].shape[0], nn_params["Theta2"].shape[1]:].transpose())
    # print(weights2.shape, sigmaO.shape)
    
    h_theta = sigmoid(sigmaO, cc, Vdd, False)                               # Line 186: nnCostFunction2.m
    h_theta = np.multiply(1 / Vdd, np.add(h_theta, Vdd / 2))                # Line 191: nnCostFunction2.m
    h_theta_pos = (h_theta == 0) * np.finfo(float).tiny + h_theta           # Line 192: nnCostFunction2.m
    h_theta_neg = 1 - h_theta                                               # Line 193: nnCostFunction2.m
    h_theta_neg = (h_theta_neg == 0) * np.finfo(float).tiny + h_theta       # Line 194: nnCostFunction2.m
    # print(h_theta.shape, h_theta_pos.shape, h_theta_neg.shape)
    
    # Calculating the cost function, which should only be executed once, otherwise J will continue to be addd to
    # Line 195: nnCostFunction2.m
    J = J + (1 / m) * np.sum(np.sum(np.multiply(1 / Vdd, np.multiply(-(Vdd / 2 + Y), np.log(h_theta_pos)) + np.multiply(-(Vdd / 2 + Y), np.log(h_theta_neg))), axis=0))
    
    # Line 199: nnCostFunction2.m
    J = J + (lbda / (2 * m)) * (np.sum(np.sum(np.square(weight_mapping(nn_params["Theta1"], K[0])), axis=0)) - \
        np.sum(np.sum(np.square(weight_mapping(nn_params["Theta1"][:, 0].reshape(-1, 1), K[0])))) + \
        np.sum(np.sum(np.square(weight_mapping(nn_params["Theta2"], K[0])), axis=0)) - \
        np.sum(np.sum(np.square(weight_mapping(nn_params["Theta2"][:, 0].reshape(-1, 1), K[0])))))
    print("Neural Network Cost: " + str(J))
    
    nn_params["a_1"] = np.concatenate((pbias.transpose(), X.transpose()), axis=0)                                   # Line 203: nnCostFunction2.m
    a_1n = np.concatenate((nbias.transpose(), -X.transpose()), axis=0)                                              # Line 204: nnCostFunction2.m
    weights1 = weight_mapping(np.concatenate((nn_params["Theta1"], nn_params["Theta1n"]), axis=1), K[0])            # Line 205: nnCostFunction2.m
    # Line 206: nnCostFunction2.m
    nn_params["z_2"] = np.matmul(weights1[:nn_params["Theta1"].shape[0], :nn_params["Theta1"].shape[1]], nn_params["a_1"]) + np.matmul(weights1[:nn_params["Theta1"].shape[0], nn_params["Theta1"].shape[1]:], a_1n)
    nn_params["a_2"] = np.concatenate((pbias.transpose(), sigmoid(nn_params["z_2"], cc, Vdd, False)), axis=0)       # Line 208: nnCostFunction2.m
    a_2n = np.concatenate((nbias.transpose(), sigmoid(nn_params["z_2"], cc2, Vdd, True)), axis=0)                   # Line 209: nnCostFunction2.m
    # print(nn_params["a_1"].shape, a_1n.shape, weights1.shape, nn_params["z_2"].shape, nn_params["a_2"].shape, a_2n.shape)
    
    # Line 297: nnCostFunction2.m
    nn_params["z_3"] = np.matmul(weights2[:nn_params["Theta2"].shape[0], :nn_params["Theta2"].shape[1]], nn_params["a_2"]) + np.matmul(weights2[:nn_params["Theta2"].shape[0], nn_params["Theta2"].shape[1]:], a_2n)
    nn_params["a_3"] = sigmoid(nn_params["z_3"], cc, Vdd, False)                # Line 299: nnCostFunction2.m
    nn_params["delta_3"] = -np.subtract(nn_params["a_3"], Y.transpose())        # Line 300: nnCostFunction2.m
    # print(nn_params["z_3"].shape, nn_params["a_3"].shape, nn_params["delta_3"].shape)
    
    # Line 301: nnCostFunction2.m
    nn_params["delta_2"] = np.multiply(np.matmul(weights2[:nn_params["Theta2"].shape[0], :nn_params["Theta2"].shape[1]].transpose(), nn_params["delta_3"]), np.concatenate((pbias.transpose(), sigmoidGradient(nn_params["z_2"], cc, False)), axis=0)) + np.multiply(np.matmul(weights2[:nn_params["Theta2"].shape[0], :nn_params["Theta2"].shape[1]].transpose(), nn_params["delta_3"]), np.concatenate((nbias.transpose(), sigmoidGradient(nn_params["z_2"], cc2, True)), axis=0))
    weights1_md = weight_md(np.concatenate((nn_params["Theta1"], nn_params["Theta1n"]), axis=1), K[3])      # Line 305: nnCostFunction2.m
    weights2_md = weight_md(np.concatenate((nn_params["Theta2"], nn_params["Theta2n"]), axis=1), K[0])      # Line 306: nnConstFunction2.m
    # print(nn_params["delta_2"].shape, weights1_md.shape, weights2_md.shape)
    
    # Extracting the values from Delta_2t every other 101 elements of the array
    # Looks like there should be a better way to do this
    Delta_2t = np.matmul(nn_params["delta_3"], np.matmul(np.concatenate((nn_params["a_2"], a_2n), axis=0).transpose(), weights2_md))       # Line 307: nnCostFunction2.m
    
    # Not sure why this needs to be declared first but after doing so, the code below works
    Delta_2n = np.zeros((nn_params["delta_3"].shape[0], nn_params["Theta2"].shape[1]))
    # Line 308: nnCostFunction2.m
    for i in range(nn_params["delta_3"].shape[0]):
        nn_params["Delta_2"][i, :] = Delta_2t[i, i * 2 * nn_params["Theta2"].shape[1]:(2 * i + 1) * nn_params["Theta2"].shape[1]]       # Line 309: nnCostFunction2.m
        Delta_2n[i, :] = Delta_2t[i, (2 * i + 1) * nn_params["Theta2"].shape[1]:(i + 1) * 2 * nn_params["Theta2"].shape[1]]             # Line 310: nnCostFunction2.m
    # print(Delta_2t.shape, nn_params["Delta_2"].shape, Delta_2n.shape)
    
    # Not sure why this needs to be declared first but after doing so, the code below works
    Delta_1n = np.zeros((nn_params["delta_2"].shape[0] - 1, nn_params["Theta1"].shape[1]))
    Delta_1t = np.matmul(nn_params["delta_2"][1:, :], np.matmul(np.concatenate((nn_params["a_1"], a_1n), axis=0).transpose(), weights1_md))     # Line 312: nnCostFunction2.m
    # Line 308: nnCostFunction2.m
    for i in range(nn_params["delta_2"].shape[0] - 1):
        nn_params["Delta_1"][i, :] = Delta_1t[i, i * 2 * nn_params["Theta1"].shape[1]:(2 * i + 1) * nn_params["Theta1"].shape[1]]       # Line 309: nnCostFunction2.m
        Delta_1n[i, :] = Delta_1t[i, (2 * i + 1) * nn_params["Theta1"].shape[1]:(i + 1) * 2 * nn_params["Theta1"].shape[1]]             # Line 310: nnCostFunction2.m
    # print(Delta_1t.shape, nn_params["Delta_1"].shape, Delta_1n.shape)
    
    nn_params["Theta2_grad"][:, 0] = (2 / Vdd) * sharp_factor * (1 / m) * nn_params["Delta_2"][:, 0]                                                    # Line 320: nnCostFunction2.m
    nn_params["Theta2_grad"][:, 1:] = (2 / Vdd) * sharp_factor * (1 / m) * nn_params["Delta_2"][:, 1:] + (lbda / m) * nn_params["Theta2"][:, 1:]        # Line 321: nnCostFunction2.m
    nn_params["Theta2n_grad"][:, 0] = (2 / Vdd) * sharp_factor * (1 / m) * Delta_2n[:, 0]                                                               # Line 323: nnCostFunction2.m
    nn_params["Theta2n_grad"][:, 1:] = (2 / Vdd) * sharp_factor * (1 / m) * Delta_2n[:, 1:] + (lbda / m) * nn_params["Theta2n"][:, 1:]                  # Line 324: nnCostFunction2.m
    # print(nn_params["Theta2_grad"].shape, nn_params["Theta2n_grad"].shape)
    
    nn_params["Theta1_grad"][:, 0] = sharp_factor * (1 / m) * nn_params["Delta_1"][:, 0]                                                    # Line 369: nnCostFunction2.m
    nn_params["Theta1_grad"][:, 1:] = sharp_factor * (1 / m) * nn_params["Delta_1"][:, 1:] + (lbda / m) * nn_params["Theta1"][:, 1:]        # Line 370: nnCostFunction2.m
    nn_params["Theta1n_grad"][:, 0] = sharp_factor * (1 / m) * Delta_1n[:, 0]                                                               # Line 372: nnCostFunction2.m
    nn_params["Theta1n_grad"][:, 1:] = sharp_factor * (1 / m) * Delta_1n[:, 1:] + (lbda / m) * nn_params["Theta1n"][:, 1:]                  # Line 373: nnCostFunction2.m
    # print(nn_params["Theta1_grad"].shape, nn_params["Theta1n_grad"].shape)


# Main function
def main():
    # Parameters for activation function from SPICE when Vdd = 1 V
    cc = np.array([-0.0012, -0.2483, -0.0235, 31.7581])             # Line 11: main.m
    # Parameters for activation function from SPICE when Vdd = 0.5 V
    cc2 = np.array([-0.0011, 0.2490, -0.0203, 193.0602])            # Line 12: main.m
    # Regularization term
    lbda = 0                    # Line 13: main.m
    # J = np.zeros((4, 3))      # Line 14: main.m
    J = 0                       # Line 78: nnCostFunction2.m
    
    # Maximum iterations to run
    max_iter = 100000           # Line 15: main.m
    sharp_factor = cc[3]        # Line 17: main.m
    sharp_factorn = cc2[3]      # Line 18: main.m
    
    rand_val = np.random.permutation(np.arange(1000))[:800]     # Line 99: main.m
    mnist = loadmat("MNIST_complete.mat")                       # Line 100: main.m
    
    X = mnist["activationsPooled"]      # Line 101: main.m
    Y = mnist["y"].transpose()          # Line 102: main.m
    X = (X[:, rand_val] - 0.5) / 2      # Line 103: main.m
    Y = (Y[:, rand_val] - 0.5) / 2      # Line 104: main.m
    
    input_layer_size = X.shape[0]       # Line 116: main.m
    num_labels = Y.shape[0]             # Line 117: main.m
    
    stop_sign1 = np.zeros((int(max_iter / 500), 1))     # Line 119: main.m
    test_error = np.zeros((int(max_iter / 500), 1))     # Line 120: main.m
    
    K = np.array([7.33, 7.33, 7.33, 7.33])      # Line 124: main.m
    num_hidden_layer = 1                        # Line 126: main.m
    hidden_layer_size = 100                     # Line 127: main.m
    hidden_layer_size2 = 8                      # Line 128: main.m
    hidden_layer_size3 = 8                      # Line 129: main.m
    Vdd = 1                                     # Line 130: main.m
    # print(X.shape, Y.shape)
    
    my_backpropagation(cc, cc2, lbda, J, sharp_factor, X, Y, input_layer_size, hidden_layer_size, num_labels, K, Vdd)


# Using an if statement to executing main function only when module is run directly
if __name__ == "__main__":
    main()