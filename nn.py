from math import *
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01

## Funcs ########################################
def sig(x):
    return 1 / (1 + np.exp(-x))

def sigd(x):
    return x*(1-x)

def err(y, yh):
    0.5 * ((abs(y - yh)) ** 2)

    
## Input ########################################

# XOR
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Repeatable random vals
np.random.seed(200)

# Synapses (weights)
s1 = np.random.uniform(size=(3,4))
s2 = np.random.uniform(size=(4,1))

## Train ########################################
print("Begin training:")
for j in range(1000000):
    # Layers (4,4) (4,1)
    l1 = sig(np.dot(X,s1))
    l2 = sig(np.dot(l1,s2))

    # Backpopagation
    l2e = (Y - l2) * lr
    l2d = l2e * sigd(l2)
    l1e = np.dot(l2d, s2.T) * lr
    l1d = l1e * sigd(l1)

    # Print first layer error
    if (j % 100000) == 0:
        e = np.mean(np.abs(l2e))
        print("Error: " + str(e))
        print("Layer 1: ")
        print(l1)
        print("Layer 2: ")
        print(l2)
        print("Syn 1: ")
        print(s1)
        print("Syn 2: ")
        print(s2)

    # Set new weights (3,4) (4,1)
    s2 += np.dot(l1.T, l2d)
    s1 += np.dot(X.T, l1d)
    
## Output ########################################
print("Solution: ")
print(l2)

def feed(X):
    l1 = sig(np.dot(X,s1))
    l2 = sig(np.dot(l1,s2))
    print(l2)
