# Single-Neuron-Neural-Network

This is a toy example that can be used to illuminate the use of gradient descent to attempt to learn a weight and bias.

Ideally, we hope that our neural networks learn fast from their errors, much like how the human mind learns greater lessons from graver mistakes. To see if this actually happens, I use the example of a single neuron with just one input. I'll train this neuron to do something ridiculously easy: take the input of 1 and output 0.

Things to note about the network (or neuron, in this case):

1. The activation function used is the sigmoid function.
2. The cost function used is the quadratic cost function.

To run the code (in Termial or any other command line):
1. Initialise the ``Network`` class:
  - e.g. net = Network(0.6, 0.9, 300)
2. call the ``train`` function:
  - e.g. train(0.15)
