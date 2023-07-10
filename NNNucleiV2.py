import tensorflow as tf
import numpy as np
from skimage.io import imread
from os import listdir


# Define helper functions
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image (number of rows)
    n_W0 -- scalar, width of an input image (number of columns)
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y


def initialize_parameters(list_of_layer_dims):
    """
    :param list_of_layer_dims -- a numpy array containing the dimensions of each layer
    array of shape [[f, f, n_C_prev, n_C_curr] [f, f, n_C_prev, n_C_curr] ... [f, f, n_C_prev, n_C_curr]]
    :return: parameters -- a dictionary of tensors containing W1, W2, ..., Wn
    """
    if isinstance(list_of_layer_dims, type(np.ndarray)):
        print("list_of_layer_dims must be a numpy array containing the dimensions of each layer")
        return
    for layer in list_of_layer_dims:
        if len(layer) != 4:
            print("each layer_dimension must be a vector of 4 elements: [f, f, n_C_prev, n_C_curr]")
            return
    num_layers = len(list_of_layer_dims)
    parameters = {}
    for l in range(1, num_layers):
        wl = 'W' + str(l)
        layer_dims = list_of_layer_dims[l-1]
        parameters[wl] = tf.get_variable(wl,
                                         layer_dims,
                                         initializer=tf.contrib.layers.xavier_initializer(seed=0))

def forward_propagation(X, parameters):
    """
    :param X: input dataset placeholder, of shape(input size, number of examples)
    :param parameters:
    :return:
    """







































