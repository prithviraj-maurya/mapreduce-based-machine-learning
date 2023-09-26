from mrjob.job import MRJob
import numpy as np


class RidgeRegression(MRJob):
    def mapper(self, _, line):
        data = line.strip().split(',')
        x = [float(i) for i in data[:-1]]
        y = float(data[-1])
        yield None, (x, y)

    def reducer_init(self):
        self.X = []
        self.y = []

    def reducer(self, _, values):
        for x, y in values:
            self.X.append(x)
            self.y.append(y)
        X = np.array(self.X)
        y = np.array(self.y)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        y_mean = np.mean(y)
        y = y - y_mean

        # Add regularization term to the diagonal of X.T @ X
        lambda_ = 0.5
        X_T_X = X.T @ X + lambda_ * np.eye(X.shape[1])

        # Calculate coefficients using the closed-form solution
        inv_X_T_X = np.linalg.inv(X_T_X)
        coeffs = inv_X_T_X @ X.T @ y
        intercept = y_mean - coeffs @ X_mean
        # yield None, (coeffs, intercept, X_mean, X_std, y_mean)
        yield None, (intercept, ",".join([str(e) for e in coeffs]))

if __name__ == '__main__':
    RidgeRegression.run()
