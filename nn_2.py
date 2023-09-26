import time
from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
from numpy import array, random, dot

class NeuralNetwork(MRJob):

    def configure_args(self):
        super(NeuralNetwork, self).configure_args()
        self.add_passthru_arg('--learning_rate', default=0.1, type=float,
                              help='learning rate')
        self.add_passthru_arg('--num_iterations', default=100, type=int,
                              help='number of iterations')
        self.add_passthru_arg('--hidden_layers', default='5,5', type=str,
                              help='number of neurons in hidden layers')
        self.add_passthru_arg('--l2_regularization', default=0.1, type=float,
                              help='L2 regularization parameter')

    def initialize_weights(self):
        # Initialize weights randomly
        self.synaptic_weights = []
        layer_sizes = [self.num_features] + \
            [int(x) for x in self.hidden_layers.split(',')] + [self.num_classes]
        for i in range(len(layer_sizes)-1):
            w = 2 * random.random((layer_sizes[i], layer_sizes[i+1])) - 1
            self.synaptic_weights.append(w)

    def activation_function(self, x):
        # Use the sigmoid activation function
        epsilon = 1e-16
        x = np.clip(x, epsilon, 1 - epsilon)
        return 1.0 / (1.0 + np.exp(-np.log(np.divide(x, 1.0 - x))))

    def activation_derivative(self, x):
        # Derivative of sigmoid function
        return x * (1 - x)

    def feedforward(self, x):
        # Calculate the output of the neural network
        activations = [x]
        for i in range(len(self.synaptic_weights)):
            dot_product = dot(activations[i], self.synaptic_weights[i])
            activation = self.activation_function(dot_product)
            activations.append(activation)
        return activations

    def backpropagation(self, x, y):
        # Calculate the errors and update the weights using backpropagation
        activations = self.feedforward(x)
        error = [y - activations[-1]]
        deltas = [error[-1] * self.activation_derivative(activations[-1])]
        for i in range(len(self.synaptic_weights)-1, 0, -1):
            error.append(dot(deltas[-1], self.synaptic_weights[i].T))
            deltas.append(error[-1] * self.activation_derivative(activations[i]))
        deltas.reverse()
        for i in range(len(self.synaptic_weights)):
            self.synaptic_weights[i] += self.learning_rate * \
                (dot(activations[i].reshape(-1, 1), deltas[i].reshape(1, -1)) +
                 self.l2_regularization * self.synaptic_weights[i])

    def mapper_init(self):
        # Initialize the neural network and other parameters
        self.learning_rate = 0.05
        self.num_iterations = 20
        self.hidden_layers = "2,2"
        self.l2_regularization = 0.01
        self.num_features = 13
        self.num_classes = 3
        self.initialize_weights()

    def mapper(self, _, line):
        # Read in a record and emit a key-value pair with the predicted class and the output of the neural network
        data = array(line.strip().split(',')[1:], dtype=float)
        label = line.strip().split(',')[0]
        output = self.feedforward(data)[-1]
        self.backpropagation(data, output)
        yield label, output.tolist()


    def reducer_init(self):
        # Initialize the counts and sums for each class
        self.counts = {}
        self.sums = {}
        self.num_classes = 3

    def reducer(self, key, values):
        # Receive the predicted class and the output of the neural network and update the counts and sums for the class
        self.counts[key] = 0
        self.sums[key] = [0.0] * self.num_classes
        for value in values:
            predicted_class = value.index(max(value))
            self.counts[key] += 1
            self.sums[key][predicted_class] += 1

    def reducer_final(self):
        # Calculate the final probabilities and emit a key-value pair with the actual class and the predicted class probabilities
        for key in self.counts.keys():
            probabilities = [x / self.counts[key] for x in self.sums[key]]
            yield key, probabilities

    def steps(self):
        return [
            MRStep(mapper_init=self.mapper_init,
                    mapper=self.mapper,
                    reducer_init=self.reducer_init,
                    reducer=self.reducer,
                    reducer_final=self.reducer_final)
        ]

if __name__ == '__main__':
    NeuralNetwork.run()


"""
Output:
This output suggests that the code has successfully run a neural network classifier using the perceptron algorithm
to classify a dataset into three classes labeled as "0.0", "1.0", and "2.0".

The output shows the class labels in the first column and the weights assigned to the two features used to classify the data in the second column.
For example, the classifier assigned a weight of 0.4791666666666667 to the first feature and 0.5208333333333334 to the second feature to classify instances
belonging to class "2.0".

Similarly, for class "1.0", the classifier assigned a weight of 0.7887323943661971 to the first feature and 0.2112676056338028 to the second feature,
and for class "0.0", the classifier assigned a weight of 0.6101694915254238 to the first feature and 0.3898305084745763 to the second feature.
"""