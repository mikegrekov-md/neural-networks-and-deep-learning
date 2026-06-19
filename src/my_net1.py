import numpy as np
import time
from utils import *
from utils import paint_output

'''
p20 examples easy gradient descent example
using 1.8,9,10,11
'''
v = np.array([0.5, -0.2, 0.1, 0.9, -0.4, 0.3, 0.7, -0.1, 0.0, 0.6])
x = np.array([1.2, 0.5, -1.0, 2.1, -0.3, 0.8, 1.5, -0.2, 0.4, 1.1])
# v = np.array([0.5])
# x = np.array([1.2])
eta = 0.01
target = 5.0
print(f"target: {target}")

y = np.dot(x, v)
print(f"y: {y}")

error = y - target

gradients = 2 * error * x
print(f"gradients: {gradients}")

print(f"error: {error}")
cost_val = (target - y) ** 2
print(f"cost: {cost_val}")

delta_v = -eta * gradients
print(f"delta_v: {delta_v}")

new_v = v + delta_v
print(f"new_v: {new_v}")

new_y = np.dot(new_v, x)
print(f"new_y: {new_y}")


def forward_propagation(v, x, eta, target, step=0, use_epsilon=False):
    y = np.dot(x, v)
    error = (y - target)
    с_old = error ** 2  # from formula 1.6
    gradients = 2 * error * x
    if use_epsilon: eta = eta / np.linalg.norm(gradients)  # normilize eta p21
    delta_v = -eta * gradients
    delta_c = np.dot(delta_v, gradients)  # by formula 1.9/1.12
    new_v = v + delta_v
    new_y = np.dot(new_v, x)
    error_new = new_y - target
    c_new = error_new ** 2
    delta_c_calc = c_new - с_old
    delta_error_by_new_old = -(error_new - error)  # ?
    print(
        f"new_y: {round(new_y, 4)}, error_new: {round(-error_new, 4)}({round(-error, 4)}), step: {step}, eta: {round(eta, 5)}, delta_c: {round(delta_c, 4)}, delta_c_(new-old): {round(delta_c_calc, 4)}, normalized_grad: {np.linalg.norm(gradients)}")
    return [new_v, new_y, round(delta_c, 4), np.linalg.norm(gradients)]


def using_hessian(x):
    hessian = 2 * np.outer(x, x)
    return hessian


'''
calculate using hessian p21
'''
hessian = using_hessian(x)
# hessian_inveted = np.linalg.inv(hessian)      # not possible to use as we have just 1 example of 10 parameters
hessian_inveted = np.linalg.pinv(hessian)  # pseudo inverse

start_time_option2 = time.time()
step_hessian_option2 = np.linalg.solve(hessian, gradients)  #
v_newton_formula = v - step_hessian_option2
y_newton = np.dot(v_newton_formula, x)

with color(RED):
    print(f"y_newton_option2: {y_newton}, time: {round(time.time() - start_time_option2, 4)}")

start_time_option1 = time.time()
step_hessian_option1 = hessian_inveted.dot(gradients)  #
v_newton_formula = v - step_hessian_option1
y_newton = np.dot(v_newton_formula, x)
with color(RED):
    print(f"y_newton_option1: {y_newton}, time: {round(time.time() - start_time_option1, 4)}\n")
# so option 1 is less perscise but faster than option 2

time_option3 = time.time()
fp = forward_propagation(v, x, eta, target)
step = 0

# while True:
#     step += 1
#     fp = forward_propagation(fp[0], x, eta, target, step)
#     # if abs(fp[2]) == 0 or step > 1000:
#     if abs(round(fp[3], 3)) == 0 or step > 1000:
#         print(f"new v: {fp[0]}, grad: {fp[3]}")
#         break

with color(RED):
    print(f"time option3 = {round(time.time() - time_option3, 4)}")

print("-" * 90)

fp = forward_propagation(v, x, eta, target, use_epsilon=True)
step = 0
# while True:
#     step += 1
#     fp = forward_propagation(fp[0], x, eta, target, step, use_epsilon=True)
#     if abs(round(fp[2], 2)) == 0 or step > 1000:
#         print(f"new v: {fp[0]}, grad: {fp[3]}")
#         break

print(f"{Fore.YELLOW}{"Example p21":=^90}")

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Excercise on page 23
make network for 10000 examples with 10 paramenters
+ with batch 32
+ one-by-one
+ all-together
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
np.random.seed(1)
samples = 40960
features = 2
batch_size = 1  # ? if I could change to 1
epochs = 60000  # some number
eta_batches = 0.05
patience = 5
w_target = np.array([[2.5], [-1.5]])

X = np.round(np.random.randn(samples, features), 2)  # 10000 x 10
target_Y = np.matmul(X, w_target)
w = np.random.randn(features, 1)  # 10

print(f">>>> X shape: {X.shape}, Y_target shape: {target_Y.shape}, w shape: {w.shape}")

# mix samples and separate with batches
def create_batches(X, target_Y, batch_size=batch_size):
    batches = []
    shaffled_indexes = np.arange(X.shape[0])  # 10000 item indexex
    np.random.shuffle(shaffled_indexes)  # shaffle indexes

    X_shaffed = X[shaffled_indexes]
    Y_target_shaffed = target_Y[shaffled_indexes]

    for i in range(0, samples, batch_size):
        X_batch = X_shaffed[i:i + batch_size]
        Y_target_batch = Y_target_shaffed[i:i + batch_size]
        batches.append([X_batch, Y_target_batch])
    return batches

def gradients(X_batch, Y_target_batch, w):
    prediction_batch = np.matmul(X_batch, w)
    # print(f">>>>X_batch: {X_batch}")
    # print(f">>>>prediction_batch: {prediction_batch}")
    error_batch = prediction_batch - Y_target_batch
    gradients_batch = (2 / len(X_batch)) * np.matmul(X_batch.T, error_batch)  # 1.19
    # print(f">>>>>gradients_batch: {gradients_batch.T}")
    return gradients_batch

def get_new_w(w_old, eta_batch, gradients_batch):
    delta_w = (eta_batch) * gradients_batch  #
    new_w = w_old - delta_w  # 1.20
    # print(f">>>>>new_w: {new_w}")
    return new_w

def batch_gradient_desc(X_batch, Y_target_batch, eta_batchs, w):
    gradients_batch = gradients(X_batch, Y_target_batch, w)
    new_w = get_new_w(w_old=w, eta_batch=eta_batchs, batch_size=X_batch.shape[0], gradients_batch=gradients_batch)
    new_prediction = np.matmul(X_batch, new_w)
    error_batch = (new_prediction - Y_target_batch)
    return new_w, error_batch

def gradient_descent_with_batches(epochs, X, target_Y, batch_size, eta_batches, w):
    w_batch = w
    best_cost = float('inf')
    best_w = None
    patience_counter = 0
    for i in range(epochs):
        batches = create_batches(X, target_Y, batch_size)
        for batch in batches:
            X_batch, Y_target_batch = batch
            gradients_batch = gradients(X_batch, Y_target_batch, w_batch)
            new_w = get_new_w(w_old=w_batch, eta_batch=eta_batches, gradients_batch=gradients_batch)
            # new_prediction = np.matmul(X_batch, new_w)
            # error_batch = (new_prediction - Y_target_batch)
            w_batch = new_w
        cost_current_epoch = np.mean((X @ w_batch - target_Y) ** 2)
        if cost_current_epoch < best_cost:
            best_cost = cost_current_epoch
            best_w = w_batch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f" На эпохе {i} ошибка перестала падать.")
            print(f"Обучение остановлено! Лучшая ошибка: {best_cost:.5f}")
            print(f"Обучение остановлено! Current w: {best_w}")
            break

    prediction_batch = X @ w_batch
    error_batch = prediction_batch - target_Y
start_time = time.time()
gradient_descent_with_batches(epochs, X, target_Y, batch_size, eta_batches, w)
end_time = time.time()
print(f">>>> execution time: {round(end_time - start_time,6)} with batch size = {batch_size}")
start_time = time.time()
gradient_descent_with_batches(epochs, X, target_Y, 20, eta_batches, w)
end_time = time.time()
print(f">>>> execution time: {round(end_time - start_time,6)} with batch size = {20}")  #time of exec to prove exc.p.23

# create batchs with every usage shaffling
# go through batches
# repeate
