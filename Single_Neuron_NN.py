import numpy as np

class Network(object):

    def __init__(self, w, b, iterations):
        """
        ``w`` is the chosen weight.
        ``b`` is the chosen bias.
        ``iterations`` is the number of iterations that the neuron will iterate over.

        In this toy example, the weight and bias of the neuron is directly related to the error of a neuron. The larger the weight and bias, the greater the error. Hence, the weight and bias are manually chosen and not randomly generated like most other networks.
        """
        self.weight = w
        self.bias = b
        self.training_inputs = [1] * iterations # Training inputs of ``1``s.
        self.training_outputs = [0] * iterations # Desired training outputs of ``0``s.
        self.training_data = [(self.training_inputs[i], self.training_outputs[i]) for i in range(iterations)]
        # Note that ``train_data`` is a list of tuples (x,y) representing the
        # training inputs and the desired outputs.


    def train(self, eta):
        for (x,y) in self.training_data:
            self.update_mini_batch(x, y, eta)


    def update_mini_batch(self, x, y, eta):
        """
        Update the neuron's weights and biases by applying gradient descent using backpropogation to a single mini batch. The ``mini_batch`` is a list of tuples (x, y), and ``eta`` is the learning rate.
        """

        nb, nw = self.backprop(x, y)
        self.weight = self.weight - (eta)*nw
        self.bias = self.bias - (eta)*nb


    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x. The input (x, y) is the ``mini_batch`` (1, 0), where
        ``1`` is the training input and ``0`` is the desired training output.
        """

        z = (self.weight * x) + self.bias

        # Feedforward:
        output_activation = self.sigmoid(z)
        print(output_activation)
        # Output error:
        delta = (output_activation - y) * self.sigmoid_prime(z)

        # Backpropagate the error:
        nabla_Cb = delta
        nabla_Cw = x * delta
        # Note: the actual formula is (input activation * delta). Since this is a single neuron network, the input activation is ``1``.

        #Output:
        return (nabla_Cb, nabla_Cw)


    def sigmoid(self, z):
        """
        The sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(self, z):
        """
        Derivative of the sigmoid function.
        """
        return sigmoid(z) * (1 - sigmoid(z))
