import numpy as np
import activation as act
import optimizers as opt

def initialize(weightsDim, biasDim):
	weights = np.array([np.random.random(row)-0.5 for row in weightsDim], dtype=object)
	bias = np.array([np.random.random(row)-0.5 for row in biasDim], dtype=object)
	return weights, bias

def cost(real, guess):
	return np.sum((real - guess) ** 2)

def forwProp(inputs, weights, bias, activation):
	outputs = []
	for row in range(len(bias)):
		inputs.append(weights[row].dot(inputs[-1]) + bias[row])
		outputs.append(activation(inputs[-1]))
	return inputs[1:], outputs

def backProp(inputs, outputs, weights, x, y, rate=0.1, activation=act.sigmoid, activationDeriv=act.sigmoidDeriv, optimizer=opt.gradientDescent):
	for row in range(1, len(outputs)+1):
		outputGrad = cost(outputs, y)
		weightGrad = outputGrad.dot(outputs[-row].T)
		biasGrad = np.sum(outputGrad)

		weight[-row] -= rate * weightGrad
		bias[-row] -= rate * biasGrad

def train(x, y, weights, bias, params):
	inputs, outputs = forwProp(x, weights, bias, activation)
	weightGrad, biasGrad = backProp(inputs, outputs, weights, x, y, optimizer, *params)

z = inputs
a = outputs

dz2 = a2 - y
dw2 = dz2.dot(a1.T)
db2 = np.sum(dz2)

## update weights and biases
w2 = w2 - learning_rate * dw2
b2 = b2 - learning_rate * db2


# hidden layer -> input layer
dz1 = w2.T.dot(dz2) * relu_deriv(z1)
dw1 = dz1.dot(inputs.T)
db1 = np.sum(dz1)

## update weights and biases
w1 = w1 - learning_rate * dw1
b1 = b1 - learning_rate * db1
