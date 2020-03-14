import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        _, _, input_channels = input_shape

        self.conv1 = ConvolutionalLayer(input_channels, conv1_channels, 3, 1)
        self.ReLU1 = ReLULayer()
        self.pool1 = MaxPoolingLayer(4, 4)

        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.ReLU2 = ReLULayer()
        self.pool2 = MaxPoolingLayer(4, 4)

        self.flat = Flattener()
        self.fc = FullyConnectedLayer(4 * conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        for i in self.params():
            self.params()[i].grad = np.zeros_like(self.params()[i].value)

        out = self.pool1.forward(self.ReLU1.forward(self.conv1.forward(X)))
        out = self.pool2.forward(self.ReLU2.forward(self.conv2.forward(out)))
        out = self.fc.forward(self.flat.forward(out))

        loss, grad = softmax_with_cross_entropy(out, y)

        grad = self.flat.backward(self.fc.backward(grad))
        grad = self.conv2.backward(self.ReLU2.backward(self.pool2.backward(grad)))
        grad = self.conv1.backward(self.ReLU1.backward(self.pool1.backward(grad)))

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        out = self.pool1.forward(self.ReLU1.forward(self.conv1.forward(X)))
        out = self.pool2.forward(self.ReLU2.forward(self.conv2.forward(out)))
        out = self.fc.forward(self.flat.forward(out))

        predictions = softmax(out)
        pred = np.argmax(predictions, axis=1)

        return pred

    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters

        result = {'conv1.W': self.conv1.W, 'conv1.B': self.conv1.B,
                  'conv2.W': self.conv1.W, 'conv2.B': self.conv1.B,
                  'fc.W': self.fc.W, 'fc.B': self.fc.B}

        return result
