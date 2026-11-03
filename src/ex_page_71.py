'''
Construct
an
example
showing explicitly
that
in a network with a sigmoid output layer, the output activations aLj won’t always sum to 1.
'''

import numpy as np
import random

# x = np.array([1, 2, 3])
my_list = [random.randint(1, 10) for _ in range(3)]
my_list = [9, 5, 3]
x = np.array(my_list)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, k = 1):
    x_with_temp = x * k
    e_x = np.exp(x_with_temp - np.max(x_with_temp))
    return e_x / e_x.sum()


def binary_cross_enropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    pass


sigmoid_ = sigmoid(x)
softmax_ = softmax(x, 1000000000)

print(f"my_list: {my_list}")
print(f"sigmoid(x) = {sigmoid_} with sum = {sigmoid_.sum()}")
print(f"softmax(x) = {softmax_} with sum = {softmax_.sum()}")

num_params_hidden = 768 * 30 + 30
num_params_output = 30 * 10 + 10
total_params = num_params_hidden + num_params_output
print(f"total_params = {total_params}")