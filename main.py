import numpy as np


class NN:
	def __init__(self,
			inputs_count,
			hidden_layers,
			outputs_count,
			activation,
			activation_der,
			learning_rate = 0.3,
			_generate = True):
		self.generate_neurons(hidden_layers, outputs_count)
		if _generate:
			self.generate_weights(inputs_count, hidden_layers, outputs_count)
			self.generate_biases(hidden_layers, outputs_count)
		self.activation = activation
		self.activation_der = activation_der
		self.learning_rate = learning_rate

	def load(path, activation, activation_der, learning_rate = 0.3):
		with open(path, 'r') as f:
			file_data = f.read().split('\n')
			inputs = int(file_data[1])
			hidden = int(file_data[2])
			outputs = int(file_data[3])
			hidden_neurons = []
			for i in range(4, 3 + hidden):
				hidden_neurons.append(int(file_data[i]))
			brain = NN(inputs,
					hidden_neurons,
					outputs,
					activation,
					activation_der,
					learning_rate,
					False)
			brain.weights = []
			for i in range(5 + hidden, 5 + 2 * hidden):
				layer_data = file_data[i].split(' ')
				layer_metadata = [int(n) for n in layer_data[0].split(',')]
				left_neurons = layer_metadata[0]
				right_neurons = layer_metadata[1]
				if right_neurons != 1:
					layer_weights_data = np.array([float(weight) for weight in layer_data[1:]], 'float').reshape(
						left_neurons, right_neurons)
				else:
					layer_weights_data = np.array([float(weight) for weight in layer_data[1:]], 'float')
				brain.weights.append(layer_weights_data)
			brain.biases = []
			for i in range(6 + 2 * hidden, 6 + 3 * hidden):
				layer_biases = [float(bias) for bias in file_data[i].split(' ')]
				brain.biases.append(np.array(layer_biases, 'float'))
			return brain


	def generate_weights(self, inputs_count, hidden_layers, outputs_count):
		self.weights = [np.random.rand(hidden_layers[0], inputs_count)]
		for i in range(1, len(hidden_layers)):
			self.weights.append(
					np.random.rand(hidden_layers[i], hidden_layers[i-1]))
		self.weights.append(
				np.random.rand(outputs_count, hidden_layers[-1]))

	def generate_neurons(self, hidden_layers, outputs_count):
		self.neurons = []
		for neuron_count in hidden_layers:
			self.neurons.append(np.zeros(neuron_count, 'float'))
		self.neurons.append(np.zeros(outputs_count, 'float'))

	def generate_biases(self, hidden_layers, outputs_count):
		self.biases = []
		for neuron_count in hidden_layers:
			self.biases.append(np.random.rand(neuron_count))
		self.biases.append(np.random.rand(outputs_count))

	def forward_prop(self, inputs):
		# forward inputs
		self.neurons[0] = self.activation(np.matmul(self.weights[0], inputs) + self.biases[0])
		# forward hiddens and output layer
		for i in range(1, len(self.neurons)):
			self.neurons[i] = self.activation(
					np.matmul(self.weights[i], self.neurons[i - 1]) + self.biases[i])
		return self.neurons[-1]


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(x):
	return sigmoid(x) * (1.0 - sigmoid(x))


train_set = [
	(0.0, 0.0, 0.0),
	(0.0, 1.0, 1.0),
	(1.0, 0.0, 1.0),
	(1.0, 1.0, 0.0)
]


if __name__ == '__main__':
	# brain = NN(inputs_count = 2,
	# 		hidden_layers = [3],
	# 		outputs_count = 1,
	# 		activation = sigmoid,
	# 		activation_der = sigmoid_der)
	# print(brain.forward_prop(np.array([1.0, 0.0])))
	brain = NN.load('xor model.txt', sigmoid, sigmoid_der)
	print(brain.weights)
	print(brain.neurons)
	print(brain.biases)
	for t in train_set:
		r = brain.forward_prop(np.array(t[:2], 'float'))
		print(f"SAMPLE: {t} -> {r} ({round(r[0])})")