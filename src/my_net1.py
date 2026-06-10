import numpy as np
'''
p20 examples easy gradient descent example
using 1.8,9,10,11
'''
v = np.array([0.5, -0.2, 0.1, 0.9, -0.4, 0.3, 0.7, -0.1, 0.0, 0.6])
x = np.array([1.2, 0.5, -1.0, 2.1, -0.3, 0.8, 1.5, -0.2, 0.4, 1.1])
eta = 0.01
target = 5.0
print(f"target: {target}")

y = np.dot(x, v)
print(f"y: {y}")

error = y - target

gradients = 2 * error * x
print(f"gradients: {gradients}")

print(f"error: {error}")
print(f"cost: {gradients}")

delta_v = -eta * gradients
print(f"delta_v: {delta_v}")

new_v = v + delta_v
print(f"new_v: {new_v}")

new_y = np.dot(new_v, x)
print(f"new_y: {new_y}")

def forward_propagation(v,x,eta,target, step = 0, use_epsilon = False):
    y = np.dot(x, v)
    error = (y - target)
    с_old = error ** 2 # from formula 1.6
    gradients = 2 * error * x
    if use_epsilon: eta = eta / np.linalg.norm(gradients) # normilize eta p21
    delta_v = -eta * gradients
    delta_c = np.dot(delta_v,gradients) #by formula 1.9/1.12
    new_v = v + delta_v
    new_y = np.dot(new_v, x)
    error_new = new_y - target
    c_new = error_new ** 2
    delta_c_calc = c_new - с_old
    delta_error_by_new_old = -(error_new - error) #?
    print(f"new_y: {round(new_y,4)}, error_new: {round(-error_new,4)}({round(-error,4)}), step: {step}, eta: {round(eta,5)}, delta_c: {round(delta_c,4)}, delta_c_(new-old): {round(delta_c_calc,4)}, normalized_grad: {np.linalg.norm(gradients)}")
    return [new_v, new_y, round(delta_c,4)]

fp = forward_propagation(v,x,eta,target)
step = 0
while True:
    step += 1
    fp = forward_propagation(fp[0],x,eta,target, step)
    if abs(fp[2]) == 0 or step > 1000:
        print(f"new v: {fp[0]}")
        break

print("-"*90)

fp = forward_propagation(v,x,eta,target, use_epsilon = True)
step = 0
while True:
    step += 1
    fp = forward_propagation(fp[0],x,eta,target, step, use_epsilon = True)
    if abs(round(fp[2],2)) == 0 or step > 1000:
        print(f"new v: {fp[0]}")
        break
