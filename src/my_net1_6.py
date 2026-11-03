import sys

import numpy as np
import time
import random

from numpy.ma.core import shape
from scipy.special import lmbda

import mnist_loader
from mnist_loader import *

# from my_net1 import samples
from utils import *
from utils import paint_output

'''
p24 class implementation
'''


class Net(object):
    def __init__(self, sizes, L2=False, lmbda=1):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self._biases = [np.random.randn(a, 1) for a in
                        sizes[1:]]  # basies starting 2 layer, to be updated during learning
        self._weights = [np.random.randn(a, x) for x, a in
                         zip(sizes[:-1], sizes[
                             1:])]  # weights starting 2 layer, to be updated during learning: rows - neurons, columns - neurons l-1
        # print(f">>>>weights({len(self._weights)}):{self._weights[0].shape}, {self._weights[1].shape} \n  {self._weights}")
        self.lmbda = lmbda
        self.L2 = L2

    def print_weights_biases(self):
        print(f">> weights matrix: \n{self._weights}")
        # print(f">> weights T: \n{self._weights[0].T}")
        print(f">> biases matrix: \n{self._biases}")
        for w, b in zip(self._weights, self._biases):
            print(f">>>>>> w: \n{w} \n<<<<< b: \n{b}\n")

    @property
    def weights(self):
        return self._weights

    def sigmoid(self, z):  # Ex p25 not clear what to do as sigma = exp function
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        # da/dz derivative for sigmoid function, using Adamar *
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, a):  # where a is input for the network
        for w, b in zip(self._weights, self._biases):
            a = self.sigmoid(np.dot(w, a) + b)  # where w(3 x 2) * a(2 x 1) = a(3 x 1)
        return a

    @log_errors
    def create_batches(self, training_data, batch_size=None):
        batches = []
        # print(f">>>>>>>>>>>>>>>>>>>>> training_data LEN: {len(training_data)}")
        # shaffled_indexes = np.arange(len(training_data))  # set array of indexes for shaffling
        # np.random.shuffle(shaffled_indexes)  # shaffle indexes
        training_data_shaffed = training_data.copy()
        random.shuffle(training_data_shaffed)
        samples = len(training_data_shaffed)
        # print(f">>>>>>>>>>>>>>>>>>>>> samples: {samples}")

        for i in range(0, samples, batch_size):
            training_data_batch = training_data_shaffed[i:i + batch_size]
            batches.append([training_data_batch])
        return batches

    @log_errors
    def create_batches_matrixs(self, training_data, batch_size=None):
        batches = []
        # print(f">>>>>>>>>>>>>>>>>>>>> training_data LEN: {len(training_data)}")
        # shaffled_indexes = np.arange(len(training_data))  # set array of indexes for shaffling
        # np.random.shuffle(shaffled_indexes)  # shaffle indexes
        training_data_shaffed = training_data.copy()
        random.shuffle(training_data_shaffed)
        samples = len(training_data_shaffed)
        # print(f">>>>>>>>>>>>>>>>>>>>> samples: {samples}")

        for i in range(0, samples, batch_size):
            training_data_batch_list = training_data_shaffed[i:i + batch_size]
            X_list, Y_list = zip(*training_data_batch_list)
            X_array = np.column_stack(X_list)
            Y_array = np.column_stack(Y_list)
            # print(f">>>> X_list: {len(X_list)}")
            # print(f">>>> X_array: \n{X_array.shape}")
            # sys.exit()
            batches.append([X_array, Y_array])
        return batches

    @log_errors
    def SGD(self, training_data, epochs, batch_size, eta, test_data=None):
        # check test data - if we have validation data put testing against dedcated test range
        # go through epochs
        #   to improve network weights and biases
        #   after each epoch run validation over test data
        if test_data is not None: n_test = len(test_data)
        n = len(training_data)

        if test_data is not None:
            correct_answers = self.evaluate(test_data)
            print(
                f"Evaluation BEFORE EPOCHS is: {correct_answers} CORRECT for n_test = {n_test}, so accuracy is {round(100 * correct_answers / n_test, 2)}%")

        for epoch in range(epochs):
            # np.random.shuffle(training_data) #shaffle input every time - is done in self.create_batches
            batches = self.create_batches(training_data, batch_size)  # create batches every epoch with shaffled data
            for batch in batches:
                self.batch_update_network(learing_data=batch, eta=eta)
            if test_data is not None:
                correct_answers = self.evaluate(test_data)
                print(
                    f"Evaluation for {epoch} epoch is: {correct_answers} CORRECT for n_test = {n_test}, so accuracy is {round(100 * correct_answers / n_test, 2)}%")
            else:
                print(f">> epoch: {epoch} passed")

    @log_errors
    def SGD_matrix(self, training_data, epochs, batch_size, eta, test_data=None):
        # check test data - if we have validation data put testing against dedcated test range
        # go through epochs
        #   to improve network weights and biases
        #   after each epoch run validation over test data
        if test_data is not None: n_test = len(test_data)
        n = len(training_data)

        # if test_data is not None:
        #     correct_answers = self.evaluate(test_data)
        #     print(
        #         f"Evaluation BEFORE EPOCHS is: {correct_answers} CORRECT for n_test = {n_test}, so accuracy is {round(100 * correct_answers / n_test, 2)}%")

        for epoch in range(epochs):
            # np.random.shuffle(training_data) #shaffle input every time - is done in self.create_batches
            batches = self.create_batches_matrixs(training_data,
                                                  batch_size)  # create batches every epoch with shaffled data
            for batch in batches:
                self.batch_update_network_matrix(learing_data=batch, eta=eta, batch_size=batch_size, n=n)
            if test_data is not None:
                correct_answers = self.evaluate(test_data)
                # print(
                #     f"Evaluation for {epoch} epoch is: {correct_answers} CORRECT for n_test = {n_test}, so accuracy is {round(100 * correct_answers / n_test, 2)}%")
            else:
                print(f">> epoch: {epoch} passed")
            print(
                f"Evaluation for {epoch} epoch is: {correct_answers} CORRECT for n_test = {n_test}, so accuracy is {round(100 * correct_answers / n_test, 2)}%")

    @log_errors
    def batch_update_network(self, learing_data, eta=None):  # data is tuples of input and expected output,
        # self._weights and self._bases are used to hold network
        grad_C_b_acc = [np.zeros(b.shape) for b in self._biases]  # init empty 0 filled matrix = biasis
        grad_C_w_acc = [np.zeros(w.shape) for w in self._weights]  # init empty 0 filled matrix = weights
        # print(f">> nabla_biases: {nabla_biases}")
        # print(f">> nabla_weights: {nabla_weights}")
        m = len(learing_data[0][
                    1])  # batch length, a very huge piece of information just 1 item: list(1item)->list(batch size 10 items)->tuple(2 items x and y)
        # print(f">>>> learning data: {m}")
        # for item in learing_data[0][1]:
        #     print(f">>>>>> item: {item}")
        for x, y in learing_data[0]:  # for each (x,y) in batch is tuple where x is imput and y is expected result
            # print(f">>>>>>>>>>>>>Y={y}, X={len(x)}")
            # update weights and biases with backpropogation
            # using Gradient Accumulator, it accumulates Gradient to optimize application performance
            each_step_nabla_C_b, each_step_nabla_C_w = self.backpropagation(x, y)
            grad_C_b_acc = [ac_nb + each_step_nb for ac_nb, each_step_nb in zip(grad_C_b_acc, each_step_nabla_C_b)]
            grad_C_w_acc = [ac_nw + each_step_nw for ac_nw, each_step_nw in zip(grad_C_w_acc, each_step_nabla_C_w)]
        # update weights anb basis using accumulated gradients
        # update according to formula 1.16, 1.17
        self._weights = [w - (eta / m) * nw for w, nw in zip(self._weights, grad_C_w_acc)]  # nw -is nabla w
        self._biases = [b - (eta / m) * nb for b, nb in zip(self._biases, grad_C_b_acc)]

    # @log_errors
    def batch_update_network_matrix(self, learing_data, eta=None, batch_size=None, n=1):
        m = len(learing_data)
        grad_C_b_acc = [np.zeros(b.shape) for b in self._biases]  # init empty 0 filled matrix = biasis
        grad_C_w_acc = [np.zeros(w.shape) for w in self._weights]  # init empty 0 filled matrix = weights
        X, Y = learing_data
        # print(f">>>>PREV>>>>> matrix W = {self._weights[0].shape}")
        # print(f">>>>PREV>>>>> matrix B = {self._biases[0].shape}")

        grad_C_b_acc, grad_C_w_acc = self.backpropagation_matrix(X, Y)  # shape of weights and biases
        # print(f">>>>>>>>> each_step_nabla_C_b = {len(each_step_nabla_C_b)}")
        # print(f">>>>>>>>> each_step_nabla_C_b 0 = {(each_step_nabla_C_b[0].shape)}")
        # print(f">>>>>>>>> each_step_nabla_C_b 1 = {(each_step_nabla_C_b[1].shape)}")
        # print(f">>>>>>>>> each_step_nabla_C_w 0 = {(each_step_nabla_C_w[0].shape)}")
        # print(f">>>>>>>>> each_step_nabla_C_w 1 = {(each_step_nabla_C_w[1].shape)}")
        if self.L2:
            self._weights = [(1 - eta * self.lmbda / n) * w - (eta / m) * nw for w, nw in
                             zip(self._weights, grad_C_w_acc)]  # nw -is nabla w
            self._biases = [b - (eta / m) * nb for b, nb in zip(self._biases, grad_C_b_acc)]
            return
        self._weights = [w - (eta / batch_size) * nw for w, nw in zip(self._weights, grad_C_w_acc)]
        self._biases = [b - (eta / batch_size) * nb for b, nb in zip(self._biases, grad_C_b_acc)]
        # print(f">>>>>>>>> modul W = {self._weights[0].shape}")

    @log_errors
    def backpropagation_matrix(self, input, target, batch_size=None):  # input = X target = Y
        nabla_C_w = [np.zeros(w.shape) for w in self._weights]
        nabla_C_b = [np.zeros(b.shape) for b in self._biases]
        activation = input
        # print(f">>>>>>>> input: {len(input)}, target: {len(target)}")
        activations = [input]
        zs = []
        for w, b in zip(self._weights, self._biases):  # go through levels
            z = np.dot(w, activation) + b
            activation = self.sigmoid(z)  # n-row array
            activations.append(activation)
            zs.append(z)
        # the last value of activation = a(L) output
        dC_da = self.cost_derivative(activations[-1], target)
        delta = dC_da * self.sigmoid_prime(zs[-1])
        # delta = dC_da * self.sigmoid_prime(activations[-1])/len(input)
        nabla_C_b[-1] = np.sum(delta, axis=1, keepdims=True)  # (nabla, batch_size) to sum all nablas through batch
        nabla_C_w[-1] = delta @ activations[-2].T  # matrix multiplication - output layer

        for l in range(2, self.num_layers):  # starting after input layer formula 2.13
            # zs - moving from -2 backward to input layer
            z = zs[-l]
            # derivative of da/dz = a * (1 - a)
            sp = self.sigmoid_prime(z)
            # dC/dz(l-1) = dC/da(l-1) * sigmoid_prime(cur layer l) = (prev layer: dz * w) * sigmoid_prime
            delta = np.matmul(self._weights[-l + 1].T, delta) * sp  # delta shape is (neurons in layer, batch_size)
            # nabla_C_b: dC/db = nabla C_b = dC/dz = delta
            nabla_C_b[-l] = np.sum(delta, axis=1, keepdims=True)  # (nabla, batch_size) to sum all nablas through batch
            # nabla_C_w: dC/dw = nabla C_w = dC/dw = dC/da(l-1 layer) * da/dz * dz/dw formula BP4
            nabla_C_w[-l] = np.matmul(delta, activations[-l - 1].T)
            # print(f"nabla_C_b: {nabla_C_b}")
        return nabla_C_b, nabla_C_w

    def backpropagation(self, input, target):
        nabla_C_w = [np.zeros(w.shape) for w in self._weights]
        nabla_C_b = [np.zeros(b.shape) for b in self._biases]

        # print(f">>>> _weights: \n{self._weights}")
        # print(f">>>> nabla_C_w: \n{nabla_C_w}")
        # print(f">>>> nabla_C_b: {nabla_C_b}")
        # print(f">>>>>>>len INPUT, target = {len(input)}, {len(target)}")

        activation = input
        activations = [input]
        zs = []
        for w, b in zip(self._weights, self._biases):
            # print(f">>>>>>>>>>>w({w.shape}): \n{w}, \n"
            #       f">>>>>>>>>>>b({len(b.shape)}): \n{b}")
            z = np.dot(w, activation) + b  # feedforward z layer by layer
            zs.append(z)
            activation = self.sigmoid(z)  # n-row array
            activations.append(activation)

        # for activation define error
        # last layer: dC/da
        dC_da = self.cost_derivative(activations[-1], target)  # for output layer L
        # last layer: dC/dz = dC/da * da/dz = delta
        delta = dC_da * self.sigmoid_prime(zs[-1])  # formula 2.8 p 45, Adamar
        # delta = dC_da * self.sigmoid_prime(activations[-1])     #formula 2.8 p 45, Adamar

        # flatted_prime = (self.sigmoid_prime(activations[-1])).flatten() #to make diag
        flatted_prime = (self.sigmoid_prime(zs[-1])).flatten()  # to make diag
        diag_prime = np.diag(flatted_prime)
        delta_mult = np.matmul(diag_prime, dC_da)
        # print(f">>>>>>>>>> equity formula 2.11: {np.allclose(delta_mult, delta)}")

        # last layer: dC/db = nabla C_b = dC/dz
        nabla_C_b[-1] = delta  # formula 2.9 for layer L-last
        # last layer: dC/dw = nabla C_w = dC/dw = dC/da(l-1 layer) * da/dz * dz/dw
        nabla_C_w[-1] = np.matmul(delta, activations[-2].T)
        for l in range(2, self.num_layers):  # starting after input layer formula 2.13
            # zs - moving from -2 backward to input layer
            z = zs[-l]
            # derivative of da/dz = a * (1 - a)
            sp = self.sigmoid_prime(z)
            # dC/dz(l-1) = dC/da(l-1) * sigmoid_prime(cur layer l) = (prev layer: dz * w) * sigmoid_prime
            delta = np.matmul(self._weights[-l + 1].T, delta) * sp
            # implementation of formula 2.12
            sp_flatted = sp.flatten()
            sp_diag = np.diag(sp_flatted)
            # delta_mult = sp_diag @ self._weights[-l + 1].T @ delta_mult # delta and delta_mult are equal

            # nabla_C_b: dC/db = nabla C_b = dC/dz = delta
            nabla_C_b[-l] = delta
            # nabla_C_w: dC/dw = nabla C_w = dC/dw = dC/da(l-1 layer) * da/dz * dz/dw formula BP4
            nabla_C_w[-l] = np.matmul(delta, activations[-l - 1].T)
            # print(f"nabla_C_b: {nabla_C_b}")

        return nabla_C_b, nabla_C_w

    def evaluate(self, test_data=None):
        # calculate output from feedforward and make a tuple with target, find the largest
        test_results = [(np.argmax(self.feedforward(x)), target) for x, target in test_data]
        # calculate number of correct answers
        return sum(int(x == target) for x, target in test_results)

    def cost_derivative(self, output, target):
        return output - target

    def evalute_accuracy(self):
        pass


# net = Net([2, 3, 1])
# net.print_weights_biases()
# start_time = time.time()
print("start data loading.....")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("END data loading.....")

# print(f"<<<<<<<<<<<>>>>>>>>>>>> training_data: {len(training_data)}")
# print(f">> loading time: {time.time() - start_time}")
# net_images = Net([784, 100, 10])
start_time = time.time()
net_images = Net([784, 100, 10], L2 = True,lmbda = 0.5)
net_images.SGD_matrix(training_data=training_data[:10000], epochs=400, batch_size=10, eta=1, test_data=test_data)
end_time = time.time()
duration_matrix = end_time - start_time
print("-" * 40)
start_time = time.time()
net_images_ordinary = Net([784, 30, 10])
net_images_ordinary.SGD(training_data=training_data, epochs=300, batch_size=10, eta=1, test_data=test_data)
end_time = time.time()
duration_ordinary = end_time - start_time
print(f"time cost matrix: {duration_matrix} seconds")
print(f"time cost one-by-one: {duration_ordinary} seconds")
# print (f"training_data: {training_data[0]}")
# print (f"test_data: {test_data[0]}")
# print(f"validation_data: {validation_data[0]}")
