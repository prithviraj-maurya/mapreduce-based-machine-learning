from mrjob.job import MRJob
import numpy as np

# Define the neural network architecture
input_layer_size = 784  # 28x28 input images
hidden_layer_size = 25
output_layer_size = 10  # 10 digits (0-9)

# Initialize the weights randomly
initial_theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
initial_theta2 = np.random.rand(output_layer_size, hidden_layer_size + 1)


# Helper functions for forward and back propagation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def forward_propagation(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def back_propagation(X, y, theta1, theta2, lambda_):
    m = X.shape[0]
    a1, z2, a2, z3, h = forward_propagation(X, theta1, theta2)

    d3 = h - y
    d2 = np.dot(d3, theta2[:, 1:]) * sigmoid_gradient(z2)

    Delta1 = np.dot(d2.T, a1)
    Delta2 = np.dot(d3.T, a2)

    theta1_grad = Delta1 / m
    theta2_grad = Delta2 / m

    # Regularization
    theta1[:, 1:] = theta1[:, 1:] - (lambda_ / m) * theta1[:, 1:]
    theta2[:, 1:] = theta2[:, 1:] - (lambda_ / m) * theta2[:, 1:]

    # Add regularization term to gradient
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (lambda_ / m) * theta1[:, 1:]
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (lambda_ / m) * theta2[:, 1:]

    return theta1_grad, theta2_grad


# MapReduce job to train the neural network
class DigitClassifier(MRJob):

    def __init__(self, *args, **kwargs):
        super(DigitClassifier, self).__init__(*args, **kwargs)
        self.input_layer_size = 784  # 28x28 input images
        self.hidden_layer_size = 25
        self.output_layer_size = 10  # 10 digits (0-9)

    def mapper(self, _, line):
        # Parse the input data
        data = line.split(',')
        X = np.array(data[:-1], dtype=float).reshape(1, -1)
        y = np.zeros((1, output_layer_size))
        y[0, int(data[-1])] = 1

        yield None, (X, y)

    def combiner(self, _, pairs):
        X_batch = []
        y_batch = []
        for X, y in pairs:
            X_batch.append(X)
            y_batch.append(y)

            # Train the network in mini-batches of 100 examples
            if len(X_batch) == 100:
                # Convert list of arrays to numpy arrays
                X_batch = np.vstack(X_batch)
                y_batch = np.vstack(y_batch)

                # Compute the gradients
                theta1_grad, theta2_grad = back_propagation(X_batch, y_batch, initial_theta1, initial_theta2, 0.1)

                # Flatten the gradients to 1D arrays
                grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))

                # Yield the gradients
                yield None, grad

                # Clear the mini-batches
                X_batch = []
                y_batch = []

                # Process any remaining examples
            if len(X_batch) > 0:
                X_batch = np.vstack(X_batch)
                y_batch = np.vstack(y_batch)
                theta1_grad, theta2_grad = back_propagation(X_batch, y_batch, initial_theta1, initial_theta2, 0.1)
                grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))
                yield None, grad

    def reducer(self, _, grads):
        # Sum the gradients from all the mappers
        total_grad = np.zeros(initial_theta1.size + initial_theta2.size)

        for grad in grads:
            total_grad += grad

        # Average the gradients and update the weights
        avg_grad = total_grad / self.mr_job_runner.counters["counters"]["combiner_calls"]

        theta1_grad = avg_grad[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size,
                                                                                    input_layer_size + 1)
        theta2_grad = avg_grad[hidden_layer_size * (input_layer_size + 1):].reshape(output_layer_size,
                                                                                    hidden_layer_size + 1)

        # Update the weights using gradient descent
        alpha = 0.1
        initial_theta1 = initial_theta1 - alpha * theta1_grad
        initial_theta2 = initial_theta2 - alpha * theta2_grad

        # Emit the new weights
        yield None, (initial_theta1.tolist(), initial_theta2.tolist())

if __name__ == '__main__':
    DigitClassifier.run()
