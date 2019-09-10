import numpy as np

training_inputs = [1] * 300
# Training inputs of ``1``s.

training_outputs = [0] * 300
# Desired training outputs of ``0``s.

training_data = [(training_inputs[i], training_outputs[i])
    for i in range(len(training_inputs))]
# Note that ``train_data`` is a list of tuples (x,y) representing the
# training inputs and the desired outputs.


def SGD(training_data, epochs, mini_batch_size, eta):
    """
    Train the neural network using mini-batch stochastic gradient descent. The ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs and the desired outputs.

    ``epochs`` is the number of epochs to train for.
    ``mini_batch_size`` is the size of the mini-batches to use when sampling.
    ``eta`` is the learning rate.
    """

    n = len(training_data)

    for j in range(epochs):
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in range(0, n, mini_batch_size)]
            # Note: we are not randomly shuffling the training inputs
            # because there is only a single input of ``1``.
            # i.e. there is no point shuffling the training inputs.
            # Note: ``mini_batches`` is a list of ``mini_batch``, where
            # ``mini_batch`` is a tuple (x,y) = (1,0).

        for mini_batch in mini_batches:
            update_mini_batch(mini_batch, eta)
            # This updates the network weights and biases according to a
            # single iteration of gradient descent, using just the
            # training data in each ``mini_batch``.


def update_mini_batch(mini_batch, eta):
    """
    Update the neuron's weights and biases by applying gradient descent using backpropogation to a single mini batch. The ``mini_batch`` is a list of tuples (x, y), and ``eta`` is the learning rate.
    """

    nabla_b = 0
    nabla_w = 0

    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y)
        nabla_b =

        weight = weight - (eta/len(mini_batch))*nw for w, nw in zip()

        bias = bias - (eta/len(mini_batch))*nb for b, nb in zip()


def backprop(x, y):
    """
    Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x. The input (x, y) is the ``mini_batch`` (1, 0), where
    ``1`` is the training input and ``0`` is the desired training output.
    """

    z = (weight * x) + bias

    # Feedforward:
    output_activation = sigmoid(z)

    # Output error:
    delta = (output_activation - y) * sigmoid_prime(z)

    # Backpropagate the error:
    nabla_Cb = delta
    nabla_Cw = x * delta
    # Note: the actual formula is (input activation * delta). Since this is a single neuron network, the input activation is ``1``.

    #Output:
    return (nabla_Cb, nabla_Cw)


def feedforward(a):
    """
    Return the output of the network is ``a`` is the input.
    """
    for
    a = sigmoid((weight * a) + bias)


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
