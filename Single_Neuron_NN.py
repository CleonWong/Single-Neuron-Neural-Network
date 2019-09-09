import numpy as np

train_inputs = np.ones((300, 1))
train_outputs = no.zeros((300, 1))


def train(train_inputs, train_outputs, weight, bias, iterations):

    for iteration in range(iterations):
        test_results = feedforward(train_inputs)

def cost_derivative(output_activations, y):
    """
    Return the vector of partial derivatives \partial C_x / \partial a for the
    output activations. Note that given a quadratic cost function, the partial derivatives is simply the output activation ``a`` - the desired output ``y``.
    """
    return output_activations, y)

def feedforward(a):
    """
    Return the output of the network is ``a`` is the input.
    """
    for
    a = sigmoid(np.dot(weight, a) + bias)


def sigmoid(z):
    """
    The sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    """
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):

    def __init__(self, w, b):
        self.weights = w
        self.biases = b

    def feedforward(self, a):
        """
        Return the output of the network is ``a`` is the input.
        """
        a = sigmoid()
