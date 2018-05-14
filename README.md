# Tensorflow-Framework-for-Implementing-DNN-using-Memristor-Arrays

In this code, we designed an improved training framework for an inverter-based memristive
neuromorphic hardware that is more research community friendly. Utilizing industry-standard
TensorFlow tools, we created a declarative programming scheme for ex-situ training that built
off of a previous Matlab training paradigm. By using directed graphs to represent and compute
neural network training, this framework allows for more parallelized and efficient computation.

## Description of files

The report written in IEEE format can be seen in the final report pdf. Within the Code Implementation folder you will find the following files. 

* Tensorflow_DNN.py - Tensorflow training framework for the MNIST data set
* memristor_DNN.py – Numpy array training framework for the MNIST data set
* MNIST_complete.mat - MNIST data set used for the above network training

### Prerequisites

This training framework requires the following modules:

* Python (both 2.7X or 3.6X work fine)
* TensorFlow (verified to work on 1.5, 1.6, and 1.7)
* Numpy

To install TensorFlow, refer to the following: https://www.tensorflow.org/install/

### Execution

Ensure the above requirements are met and the python script and MNIST dataset are in the
same directory. Then, run the command:

```
$> python Tensorflow_DNN.py
```
For our Numpy array framework, run the command:

```
$> python memristor_DNN.py
```
To visualize the TensorFlow graph, run the command:
```
$> tensorboard --logdir=/tmp/tensorflow_logs
```
Then go to your browser and type http://0.0.0.0:6006 into the URL.


# CONTRIBUTORS

* Andrew McLean - amclean@usc.edu
* Alvin Wong - alvindwo@usc.edu
* Kody Ferguson - kgfergus@usc.edu
* Arash Fayyazi - fayyazi@usc.edu
