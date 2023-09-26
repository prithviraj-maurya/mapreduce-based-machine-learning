import numpy as np
from mrjob.job import MRJob


class LogisticRegressionMapReduce(MRJob):

    def __init__(self, *args, **kwargs):
        super(LogisticRegressionMapReduce, self).__init__(*args, **kwargs)
        self.num_features = 9
        self.num_iterations = 100
        self.learning_rate = 0.1

    def mapper(self, _, line):
        data = line.strip().split(',')
        # Extract the target variable
        target = int(data[-1])
        # Extract the features
        features = data[:-1]
        yield 0, (features, target)


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def reducer(self, key, values):
        # Initialize the model parameters
        w = np.zeros(len(next(values)[0]))

        # Set the learning rate and number of iterations
        learning_rate = 0.01
        num_iterations = 10
        data = []
        # Update the model parameters using gradient descent
        for i in range(num_iterations):
            for features, target in values:
                features = [float(x) for x in features]
                features = np.array(features, dtype=np.float64)
                target = int(target)
                data.append((features, target))
                z = np.dot(w, features)
                z = np.clip(z, -500, 500)
                h = self.sigmoid(z)
                temp = learning_rate * (target - h)
                w += temp * features

        num_correct = 0
        for features, target in data:
                features = [float(x) for x in features]
                features = np.array(features, dtype=np.float64)
                target = int(target)
                z = np.dot(w, features)
                z = np.clip(z, -500, 500)
                h = self.sigmoid(z)
                predicted_target = 1 if h >= 0.5 else 0
                if predicted_target == target:
                    num_correct += 1

        accuracy = num_correct / len(data)

        yield None, accuracy


if __name__ == '__main__':
    LogisticRegressionMapReduce.run()
