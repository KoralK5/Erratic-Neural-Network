import numpy as np

def ReLU(x):
	return np.maximum(0, x)

def ReLUderiv(x):
	return x > 0

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoidDeriv(x):
	return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))
