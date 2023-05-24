# Backpropagation for XOR Network

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def xor(inputs):
    return np.array([int(inputs[0] != inputs[1])])


input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]])
target_data = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])

input_size = 2
hidden_size = 8
output_size = 1

# initialize the weights with random values
hidden_weights = np.random.uniform(size=(input_size, hidden_size))
output_weights = np.random.uniform(size=(hidden_size, output_size))

# define the learning rate and number of epochs
learning_rate = 0.1
epochs = 100000

# train the neural network using backpropagation
for epoch in range(epochs):
    # forward propagation
    hidden_layer = sigmoid(np.dot(input_data, hidden_weights))
    output_layer = sigmoid(np.dot(hidden_layer, output_weights))

    # backward propagation
    output_error = target_data - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)

    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    # updating
    output_weights += hidden_layer.T.dot(output_delta) * learning_rate
    hidden_weights += input_data.T.dot(hidden_delta) * learning_rate

# testing
test_input = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 1]])
for i in range(len(test_input)):
    prediction = sigmoid(np.dot(sigmoid(np.dot(test_input[i], hidden_weights)), output_weights))
    print(f"Input: {test_input[i]} Output: {prediction.round()} Target: {xor(test_input[i])}")