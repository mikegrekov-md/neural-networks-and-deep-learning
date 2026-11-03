'''
page 34 SVC checking, takes x3 time longer
then a network
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import time

import mnist_loader
# from mnist_loader import *

# from my_net1 import samples
from utils import *
from utils import paint_output

print("start data loading.....")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(vectorized=False)
print("END data loading.....")


training_x = np.array([x[0] for x in training_data])[:,:,0]
training_y = np.array([y[1] for y in training_data])

validation_x = np.array([x[0] for x in validation_data])[:,:,0]
validation_y = np.array([y[1] for y in validation_data])

# training_x, training_y = training_data_np[:0], training_data_np[:,1]
# training_x = training_data[:,0]
# training_y = training_data[:,1]
# print("END data converting.....")

print(f"training data x: {training_x.shape}")
print(f"training data y: {training_y.shape}")
print(f"validation data x: {validation_x.shape}")
print(f"validation data y: {validation_y.shape}")

#scaling....
print("start data scaling.....")
scaler = StandardScaler()
scaler.fit(training_x)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(training_x)
x_test_scaled = scaler.transform(validation_x)
print("END data scaling.....")
#...scaled
#svc training
print("SVC training data ...")
start = time.time()
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(x_train_scaled, training_y)
print(f"END SVC training data ({round(time.time() - start, 2)} seconds)")
#svc verification
print("SVC validation data")
svc_predictions = clf.predict(x_test_scaled)
#svc results
# print(f"SVC predictions: {svc_predictions}")
print(metrics.classification_report(validation_y, svc_predictions))