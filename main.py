import numpy as np
import NeuralNetwork as nn
import activation as act
import optimizers as opt

weightsDim = [(18, 784), (16, 10)]
biasDim = [(18, 1), (16, 1)]

inputs = np.random.rand(100, 784)
outputs = np.random.rand(100, 10)
weights, bias = nn.initialize(weightsDim, biasDim)

epochs = 3
iters = len(outputs)
rate = 0.1

activation = act.sigmoid
activationDeriv = act.sigmoidDeriv

optimizer = opt.gradientDescent

params = (rate, activation, activationDeriv)

for epoch in range(epochs):
	for row in range(iters):
		weights, bias = train(x[row], y[row], weights, bias, *params)
