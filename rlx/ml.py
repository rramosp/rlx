import numpy as np

def abs_error(estimator, X, y):
    preds = estimator.predict(X)
    return np.mean(np.abs(y-preds))


class BaselinePredictor:
    """
    chooses the source column with the values closes to the prediction
    """

    def fit(self, X, y):
        self.col = np.argmin([np.mean(np.abs(X[:, i] - y)) for i in range(X.shape[1])])

    def predict(self, X):
        return X[:, self.col]

    def score(self, X, y):
        return abs_error(self, X,y)