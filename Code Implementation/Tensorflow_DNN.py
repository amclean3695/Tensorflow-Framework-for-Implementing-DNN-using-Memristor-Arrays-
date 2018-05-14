# Neuromorphic NN Ex-Situ Training
# EE599 Spring 2018
# Andrew McClean, Alvin Wong, Kody Ferguson

import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def weight_mapping(theta,spice_params):
    '''Function that implements the weight mapping function from weight_mapping.m
    Parameters
    ----------
    theta := unconstrained weight matrix
    K := 𝐾 is a constant number which is the difference between the 𝜎𝑚𝑎𝑥 and 𝜎𝑚𝑖𝑛.
    Results
    ---------
    returns constrained mapped weight matrix
    ''' 
    theta = tf.add(tf.constant(1.0),tf.div(spice_params['K'], tf.add(tf.constant(1.0), tf.exp(-theta))))
    sum_theta = tf.reduce_sum(theta, 1, keepdims = True)
    multiples = theta.get_shape().as_list()[1] 
    return tf.div(theta, tf.tile(sum_theta, [1, multiples]))


def randInitializeWeights(L_in,L_out):
    '''Function that randomly initializes the weights of a layer with L_in incoming connections
    and L_out outgoing connections as implemented in randInitializeWeights.m
    Note that W should be set to a matrix of size(L_out, 1 + L_in) as the column row of W handles 
    the "bias" terms. Duplicates functionality from randInitializeWeights.m 
    Parameters
    ----------
    L_in := tensor, number of incoming connections
    L_out := tensor, number of outgoing connections
    Results
    returns a tensor with of initial weights for the network
    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    #
    # Note: The first row of W corresponds to the parameters for the bias units
    #
    # Randomly initialize the weights to small values
    '''
    epsilon = np.sqrt(6/(L_in+L_out))
    return tf.random_uniform([L_out, L_in + 1], dtype = tf.float32, minval = -np.sqrt(6/(L_in+L_out)), maxval = np.sqrt(6/(L_in+L_out)))


def sigmoidGradient(z, spice_params, nmos):
    '''Calculates the sigmoid gradient activation function at z, this function combines the sigmoidGradient.m and sigmoidGradientn.m 
    functions in Matlab, considering they are very similar to eachother and can be combined by adding an extra argument.
    Parameters
    ----------
    z := net inputs into the neurons in a specific layer
    cc := list, activation parameters taken from hspice
    nmos := boolean, determines whether to implement sigmoidGradient.m or sigmoidGradientn.m functionality (False, True respectively)
    Results
    ---------
    returns activated neuron output
    ''' 
    if nmos:
        sharp_factor = tf.constant(spice_params['cc2'][3], dtype = tf.float32)
        s_param = spice_params['cc2'][2]
    else:
        sharp_factor = tf.constant(-spice_params['cc'][3], dtype = tf.float32)
        s_param = spice_params['cc'][2]
    
    return tf.multiply(sharp_factor,tf.subtract(1.0,tf.square(tf.tanh(tf.multiply(sharp_factor,tf.subtract(z,s_param))))))   


def sigmoid(z,spice_params,nmos):
    '''Function that computes the sigmoid from sigmoid.m
    Parameters
    ----------
    z := weights, for calculating the activation shape(800,100)
    spice_params := dictionary of parameters
    nmos := bool, if we are computing the sigmoid for nmos
    Results
    returns a tensor after the activation of weights
    '''
    # Implement Sigmoidn.m function
    if nmos:   
        halfVdd = spice_params['Vdd']/2.0
        sharp_factor = tf.constant(spice_params['cc2'][3], dtype = tf.float32)
        s_param = spice_params['cc2'][2]
    else:
        halfVdd = -spice_params['Vdd']/2.0 
        sharp_factor = tf.constant(spice_params['cc'][3], dtype = tf.float32)
        s_param = spice_params['cc'][2]
        
    return tf.multiply(halfVdd, tf.tanh(tf.multiply(sharp_factor, tf.subtract(z, s_param))))


def weight_md(theta,spice_params):
    '''Implements physical circuit constraints on the weight mapping that occurs during backpropogation of the network. This
    function duplicates the functionality of the weight_md.m file.
    Parameters
    ----------
    theta := unconstrained weights matrix
    K := 𝐾 is a constant number which is the difference between the 𝜎𝑚𝑎𝑥 and 𝜎𝑚𝑖𝑛. 
    Results
    ---------
    returns activated neuron output
    ''' 
    num_pn = theta.get_shape().as_list()[0]
    num_n = theta.get_shape().as_list()[1]
    g1 = tf.multiply(spice_params['K'],tf.multiply(tf.divide(1.0,tf.add(1.0,tf.exp(-theta))),                                                    tf.subtract(1.0,tf.divide(1.0,tf.add(1.0,tf.exp(-theta))))))
    g1 = tf.transpose(tf.contrib.kfac.utils.kronecker_product(g1,tf.ones((num_n, 1))))
    theta = tf.add(1.0, tf.divide(spice_params['K'],tf.add(1.0,tf.exp(-theta))))
    sum_theta = tf.reduce_sum(theta, 1, keepdims = True)
    sums_theta1 = tf.tile(sum_theta,(1,num_n))
    theta = tf.transpose(tf.contrib.kfac.utils.kronecker_product(theta,tf.ones((num_n, 1))))
    sums_theta = tf.transpose(tf.contrib.kfac.utils.kronecker_product(sums_theta1,tf.ones((num_n, 1))))
    
    # Didn't need to do the extra transpose 
    sums_theta1 = tf.divide(1.0,tf.reshape(sums_theta1, [-1]))
    
    # Calculates the indexes we need from the g array 
    indices = [[i%num_n,i] for i in range(num_n*num_pn)]

    g = tf.divide(-theta, tf.square(sums_theta))
    
    # The updated weights
    updated_weights = tf.add(tf.gather_nd(g, indices), sums_theta1)

    # Indicates a sparse tensor 
    delta = tf.SparseTensor(indices, updated_weights, g.get_shape().as_list())
    
    g = tf.sparse_add(g,delta)
   
    g = tf.multiply(g,g1)
    
    return g 


def nnCostFunction(J, nn_params):
    
    # Initializes the delta of weights and biases of network
    Delta_2 = tf.zeros(tf.shape(nn_params['Theta2']))
    Delta_1 = tf.zeros(tf.shape(nn_params['Theta1']))
    delta_3 = tf.zeros([num_labels,1])
    delta_2 = tf.zeros([hidden_layer_size,1])
    
    # Maps the unconstrained weights to the weights used in training of ex-situ network
    weights1 = weight_mapping(tf.concat([nn_params['Theta1'],nn_params['Theta1n']], 1), spice_params)
    weights2 = weight_mapping(tf.concat([nn_params['Theta2'],nn_params['Theta2n']], 1), spice_params)
    
    # Forward propogating through with the weights and inputs of the mapped network weights 
    sigma1 = tf.add(tf.matmul(X1,tf.transpose(weights1[:tf.shape(nn_params['Theta1'])[0],:tf.shape(nn_params['Theta1'])[1]])),                 tf.matmul(X2,tf.transpose(weights1[:tf.shape(nn_params['Theta1'])[0],tf.shape(nn_params['Theta1'])[1]:])))
    h1 = sigmoid(sigma1,spice_params,False)

    sigmaO = tf.add(tf.matmul(tf.concat([pbias,h1],1),tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],:tf.shape(nn_params['Theta2'])[1]])),                 tf.matmul(tf.concat([nbias,sigmoid(sigma1,spice_params,True)],1),tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],tf.shape(nn_params['Theta2'])[1]:])))
    
    # Output of the weights from last activation layer
    with tf.name_scope('H_theta'):
        h_theta = sigmoid(sigmaO, spice_params, False)
    h_theta_pos = tf.multiply((1/spice_params['Vdd']), tf.add(h_theta, spice_params['Vdd']/2.0))
    h_theta_neg = tf.subtract(1.0, h_theta_pos)
    
    # Computes the cost related to this pass through the network 
    sum_theta = tf.reduce_sum(tf.reduce_sum(tf.multiply(1/spice_params['Vdd'], tf.add(tf.multiply(-(tf.add((spice_params['Vdd']/2.0),Y)),            tf.log(h_theta_pos)) ,tf.multiply(tf.add((-spice_params['Vdd']/2.0),Y),tf.log(h_theta_neg)))), 0))
    J = tf.add(J, tf.multiply(tf.cast(tf.divide(1,m), dtype = tf.float32), sum_theta))

    sum_weights = tf.reduce_sum(tf.reduce_sum(tf.square(weight_mapping(nn_params['Theta1'], spice_params)), 1, keepdims = True)) -         tf.reduce_sum(tf.reduce_sum(tf.square(weight_mapping(tf.reshape(nn_params['Theta1'][:,0], (-1,1)),spice_params)), 1, keepdims = True)) +         tf.reduce_sum(tf.reduce_sum(tf.square(weight_mapping(nn_params['Theta2'], spice_params)), 1, keepdims = True)) -         tf.reduce_sum(tf.reduce_sum(tf.square(weight_mapping(tf.reshape(nn_params['Theta2'][:,0], (-1,1)),spice_params)), 1, keepdims = True))

    J = tf.add(J, tf.multiply(tf.divide(spice_params['lambda']/2,tf.cast(tf.multiply(2,m), dtype = tf.float32)),sum_weights))

    # Forward Propogation Step
    a_1 = tf.concat([tf.transpose(pbias), tf.transpose(X)], 0)
    a_1n = tf.concat([tf.transpose(nbias), tf.transpose(-X)], 0)
    z_2 = tf.add(tf.matmul(weights1[:tf.shape(nn_params['Theta1'])[0],:tf.shape(nn_params['Theta1'])[1]], a_1),
                    tf.matmul(weights1[:tf.shape(nn_params['Theta1'])[0],tf.shape(nn_params['Theta1'])[1]:], a_1n))
    a_2 = tf.concat([tf.transpose(pbias), sigmoid(z_2, spice_params, False)], 0)
    a_2n = tf.concat([tf.transpose(nbias), sigmoid(z_2, spice_params, True)], 0)
    z_3 = tf.add(tf.matmul(weights2[:tf.shape(nn_params['Theta2'])[0],:tf.shape(nn_params['Theta2'])[1]],a_2),
                    tf.matmul(weights2[:tf.shape(nn_params['Theta2'])[0],tf.shape(nn_params['Theta2'])[1]:],a_2n))
    a_3 = sigmoid(z_3, spice_params, False)
    
    # Back Propogation Step
    delta_3 = -tf.subtract(a_3, tf.transpose(Y))
    delta_2 = tf.add(tf.multiply(tf.matmul(tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],:tf.shape(nn_params['Theta2'])[1]]), delta_3),                                         tf.concat([tf.transpose(pbias), sigmoidGradient(z_2, spice_params, False)], 0)),                            tf.multiply(tf.matmul(tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],:tf.shape(nn_params['Theta2'])[1]]), delta_3),                                         tf.concat([tf.transpose(nbias), sigmoidGradient(z_2, spice_params, True)], 0)))    
    
    # Re-Mapping the unconstrianed weights 
    weights1_md = weight_md(tf.concat([nn_params['Theta1'], nn_params['Theta1n']], 1), spice_params)
    weights2_md = weight_md(tf.concat([nn_params['Theta2'], nn_params['Theta2n']], 1), spice_params)

    # Total delta from the 2nd layer of network
    Delta_2t = tf.matmul(delta_3, tf.matmul(tf.transpose(tf.concat([a_2,a_2n], 0)), weights2_md))

    # Calculates indices to gather values from total delta of 2nd layer for positive and negative contributions respectively
    Delta2_pidx = []
    Delta2_nidx = []
    Delta2_row = delta_3.get_shape()[0]
    Delta2_col = nn_params['Theta2'].get_shape().as_list()[1]

    for r in range(Delta2_row):
        #print('Positive Indices -> ROW: %s Column Range: %d - %d' % (r, r*2*col_size, (2*r+1)*col_size))
        #print('Negative Indices -> ROW: %s Column Range: %d - %d' % (r, (2*r+1)*col_size, (r+1)*2*col_size))
        Delta2_pidx.extend([[r,i] for i in range(r*2*Delta2_col, (2*r+1)*Delta2_col)])
        Delta2_nidx.extend([[r,i] for i in range((2*r+1)*Delta2_col, (r+1)*2*Delta2_col)]) 

    # Separates the contributed delta from the positive crossbar and negative crossbar of the 2nd layer
    Delta_2 = tf.reshape(tf.gather_nd(Delta_2t, Delta2_pidx),[Delta2_row, -1])
    Delta_2n = tf.reshape(tf.gather_nd(Delta_2t, Delta2_nidx),[Delta2_row, -1])
    
    # Total delta from the 1st layer of network
    Delta_1t = tf.matmul(delta_2[1:,:],tf.matmul(tf.transpose(tf.concat([a_1,a_1n], 0)), weights1_md))

    # Calculates indices to gather values from total delta of 1st layer for positive and negative contributions respectively
    Delta1_pidx = []
    Delta1_nidx = []
    Delta1_row = hidden_layer_size
    Delta1_col = nn_params['Theta1'].get_shape().as_list()[1]

    for r in range(Delta1_row):
        #print('Positive Indices -> ROW: %s Column Range: %d - %d' % (r, r*2*Delta1_col, (2*r+1)*Delta1_col))
        #print('Negative Indices -> ROW: %s Column Range: %d - %d' % (r, (2*r+1)*Delta1_col, (r+1)*2*Delta1_col))
        Delta1_pidx.extend([[r,i] for i in range(r*2*Delta1_col, (2*r+1)*Delta1_col)])
        Delta1_nidx.extend([[r,i] for i in range((2*r+1)*Delta1_col, (r+1)*2*Delta1_col)])
    
    # Separates the contributed delta from the positive crossbar and negative crossbar of the 1st layer
    Delta_1 = tf.reshape(tf.gather_nd(Delta_1t, Delta1_pidx),[Delta1_row, -1])
    Delta_1n = tf.reshape(tf.gather_nd(Delta_1t, Delta1_nidx),[Delta1_row, -1]) 
    
    # Theta 2 gradient Variables are updated
    theta_coef = tf.multiply(tf.cast(tf.divide(1,m), dtype = tf.float32), tf.constant(spice_params['cc'][3], dtype = tf.float32))

    theta2_grad = tf.concat([tf.reshape(tf.multiply(tf.multiply((2.0/spice_params['Vdd']),theta_coef), Delta_2[:,0]), [-1,1]),                          tf.add(tf.multiply(tf.multiply((2.0/spice_params['Vdd']),theta_coef), Delta_2[:,1:]),                    tf.multiply(nn_params['Theta2'][:,1:], tf.cast(tf.divide(spice_params['lambda'],m), dtype = tf.float32)))], 1)
    theta2n_grad = tf.concat([tf.reshape(tf.multiply(tf.multiply((2.0/spice_params['Vdd']),theta_coef), Delta_2n[:,0]), [-1,1]),                          tf.add(tf.multiply(tf.multiply((2.0/spice_params['Vdd']),theta_coef), Delta_2n[:,1:]),                    tf.multiply(nn_params['Theta2n'][:,1:], tf.cast(tf.divide(spice_params['lambda'],m), dtype = tf.float32)))], 1)
    
    # Theta 1 gradient Variables are updated
    theta1_grad = tf.concat([tf.reshape(tf.multiply(theta_coef, Delta_1[:,0]), [-1,1]),                          tf.add(tf.multiply(theta_coef, Delta_1[:,1:]),                    tf.multiply(nn_params['Theta1'][:,1:], tf.cast(tf.divide(spice_params['lambda'],m), dtype = tf.float32)))], 1)
    theta1n_grad = tf.concat([tf.reshape(tf.multiply(theta_coef, Delta_1n[:,0]), [-1,1]),                          tf.add(tf.multiply(theta_coef, Delta_1n[:,1:]),                    tf.multiply(nn_params['Theta1n'][:,1:], tf.cast(tf.divide(spice_params['lambda'],m), dtype = tf.float32)))], 1)
    
    return J,[theta1_grad, theta1n_grad, theta2_grad, theta2n_grad]


def predict(nn_params):
    
    # Maps the unconstrained weights to the weights used in training of ex-situ network
    weights1 = weight_mapping(tf.concat([nn_params['Theta1'],nn_params['Theta1n']], 1), spice_params)
    weights2 = weight_mapping(tf.concat([nn_params['Theta2'],nn_params['Theta2n']], 1), spice_params)
    
    # Forward propogating through with the weights and inputs of the mapped network weights 
    sigma1 = tf.add(tf.matmul(X1,tf.transpose(weights1[:tf.shape(nn_params['Theta1'])[0],:tf.shape(nn_params['Theta1'])[1]])),                 tf.matmul(X2,tf.transpose(weights1[:tf.shape(nn_params['Theta1'])[0],tf.shape(nn_params['Theta1'])[1]:])))
    h1 = sigmoid(sigma1,spice_params,False)

    sigmaO = tf.add(tf.matmul(tf.concat([pbias,h1],1),tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],:tf.shape(nn_params['Theta2'])[1]])),                 tf.matmul(tf.concat([nbias,sigmoid(sigma1,spice_params,True)],1),tf.transpose(weights2[:tf.shape(nn_params['Theta2'])[0],tf.shape(nn_params['Theta2'])[1]:])))
    
    return sigmaO


# Dictionary that holds all the Hspice params necessary for this network
with tf.name_scope('spice_params'):
    spice_params = {'K': 7.33,
                   'Vdd': 1.0,
                   'cc': np.array([-0.0012, -0.2483,  -0.0235,  31.7581]),
                   'cc2': np.array([-0.0011,  0.2490,  -0.0203, 193.0602]),
                   'lambda': 0,
                   'Ron': 100,
                   'Roff': 16e3
                   }

# Loading the image input files needed to feed into the network
train_vals = np.random.permutation(np.arange(1000))[:800] # Line 99: main.m
test_vals = np.array([i for i in range(1000) if i not in train_vals])
mnist = loadmat('MNIST_complete.mat') # Line 100: main.m

X_dat = mnist['activationsPooled'] # Line 101: main.m
Y_dat = mnist['y'].transpose() # Line 102: main.m

with tf.name_scope('train_data'):
    # Values to Train the network
    X_train = ((X_dat[:,train_vals] - 0.5 ) /2.0).transpose() # Line 103: main.m
    Y_train = ((Y_dat[:,train_vals]  - 0.5) / 2.0).transpose() # Line 104: main.m

with tf.name_scope('test_data'):
    #Values to Test the network
    X_test = ((X_dat[:,test_vals] - 0.5 ) /2.0).transpose() # Line 103: main.m
    Y_test = ((Y_dat[:,test_vals]  - 0.5) / 2.0).transpose() # Line 104: main.m

with tf.name_scope('network_params'):
    # Network Parameters
    input_layer_size = 196 # MNIST data input (img shape: 14*14)
    num_labels = 10 # MNIST total classes (0-9 digits)
    num_hidden_layer = 1 # Line 126: main.m (number of hidden layers)
    hidden_layer_size = 100 # Line 127: main.m (1st layer number of neurons)


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# tf Graph Input
with tf.name_scope('Inputs'):
    X = tf.placeholder(tf.float32, [None, 196], name='X') # mnist data image of shape 14*14 = 196
    Y = tf.placeholder(tf.float32, [None, 10], name='Y') # 0-9 digits recognition => 10 classes

with tf.name_scope('Cost_init'):
    # Initializing the Cost J
    cost = tf.Variable(tf.zeros([1,]))

# Get the number of rows in the fed value at run-time (for our case 800)
m = tf.shape(X)[0]

# Defines the positive and negative bias terms respectively (pbias,nbias)
pbias = tf.fill((m, 1),(spice_params['Vdd']/2.0))
nbias = tf.fill((m, 1),(-spice_params['Vdd']/2.0))

X1 = tf.concat([pbias, X], 1)
X2 = tf.concat([nbias, -X], 1)


# Defining the unconstrained weights
with tf.name_scope('NN_params'):
    nn_params = {
        'Theta1': tf.Variable(tf.zeros([hidden_layer_size, input_layer_size + 1])),
        'Theta1n': tf.Variable(tf.zeros([hidden_layer_size, input_layer_size + 1])),
        'Theta2': tf.Variable(tf.zeros([num_labels, hidden_layer_size + 1])),
        'Theta2n': tf.Variable(tf.zeros([num_labels, hidden_layer_size + 1])),
    }
for key in nn_params:
    print(key, nn_params[key].get_shape().as_list())


# This is the start of the NNCOSTFUNCTION 
with tf.name_scope('graph_params'):
    # Graph Parameters
    learning_rate = 0.01
    training_epochs = 100
    display_step = 1

# Data Visualization Path
logs_path = '/tmp/tensorflow_logs/'

# This is useful when we want to test the predictions
# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Predict'):
    # Model
    pred = predict(nn_params)
with tf.name_scope('Cost_Fx'):
    # Minimize error using cross entropy
    cost, gradients = nnCostFunction(cost, nn_params)
    # Computes the cost and the gradients according to the above function

# Create a summary to monitor cost tensor
tf.summary.histogram("Cost", cost)

# Random Initialization of the weights op
with tf.name_scope('Init_step'):
    init_step = [
        tf.assign(nn_params['Theta1'], randInitializeWeights(input_layer_size, hidden_layer_size)),
        tf.assign(nn_params['Theta1n'], randInitializeWeights(input_layer_size, hidden_layer_size)),
        tf.assign(nn_params['Theta2'], randInitializeWeights(hidden_layer_size, num_labels)),
        tf.assign(nn_params['Theta2n'], randInitializeWeights(hidden_layer_size, num_labels))  
    ]
# Training op which updates the weights based on our backpropogation 
with tf.name_scope('Train_step'):
    train_step = [
        tf.assign(nn_params['Theta1'], nn_params['Theta1']- learning_rate * gradients[0]),
        tf.assign(nn_params['Theta1n'], nn_params['Theta1n'] - learning_rate * gradients[1]),
        tf.assign(nn_params['Theta2'], nn_params['Theta2'] - learning_rate* gradients[2]),
        tf.assign(nn_params['Theta2n'], nn_params['Theta2n'] - learning_rate * gradients[3])  
    ]
# Merge ops for data visualization
merged = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init_step)
    # op to write logs to Tensorboard
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        # Runs a Batch Run
        _ , c, summary = sess.run([train_step, cost, merged], feed_dict = {X: X_train, Y : Y_train})
        #summary = sess.run(merged, feed_dict = {X: X_train, Y : Y_train})
        # Write logs at every iteration
        writer.add_summary(summary, epoch)
        print ('Epoch: %04d ' % (epoch+1), "cost = %0.6f" % (c))
  
    writer.close()
    print ("Optimization Finished!")
    print(nn_params['Theta2n'].eval())

