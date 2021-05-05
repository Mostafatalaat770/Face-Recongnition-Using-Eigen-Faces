import numpy as np


class PCA:

    def __init__(self, alpha):
        self.alpha = alpha
        self.components = None
        self.mean = None
        self.n_components = None
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # covariance, function needs samples as columns
        cov = np.cov(X.T)
        # eigenvalues, eigenvectors
        if not self.eigenvalues:
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov)
            # -> eigenvector v = [:,i] column vector, transpose for easier calculations
            # sort eigenvectors
            self.eigenvectors = self.eigenvectors.T
            idxs = np.argsort(self.eigenvalues)[::-1]
            self.eigenvalues = self.eigenvalues[idxs]
            self.eigenvectors = self.eigenvectors[idxs]
            # find number of n_components satisfying alpha
        self.n_components = 1
        while (self.n_components):
            f = sum(self.eigenvalues[:self.n_components]) / sum(self.eigenvalues)
            if f < self.alpha:
                self.n_components += 1
            else:
                break
        # store first n eigenvectors (projection matrix)
        self.components = self.eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        X = X - np.mean(X, axis=0)
        return np.dot(X, self.components.T)
