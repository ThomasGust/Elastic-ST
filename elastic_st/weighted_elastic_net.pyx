import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cdef class WeightedElasticNet:
    cdef:
        double alpha  # Regularization strength
        double l1_ratio  # L1 to L2 regularization ratio
        object weights  # Declare as generic object for class-level storage
        object beta  # Declare as generic object for class-level storage
        int max_iter  # Maximum iterations
        double tol  # Convergence tolerance

    def __init__(self, double alpha, double l1_ratio, cnp.ndarray[double, ndim=1] weights, int max_iter=1000, double tol=1e-4):
        """
        Initialize the Weighted Elastic Net model.

        Parameters:
            alpha (float): Regularization strength.
            l1_ratio (float): Ratio of L1 to L2 penalty.
            weights (array): Feature-specific weights.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, cnp.ndarray[double, ndim=2] X, cnp.ndarray[double, ndim=1] y):
        """
        Fit the Weighted Elastic Net model using coordinate descent.

        Parameters:
            X (array): Feature matrix (n_samples x n_features).
            y (array): Target vector (n_samples,).
        """
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        cdef cnp.ndarray[double, ndim=1] residual
        cdef cnp.ndarray[double, ndim=1] beta_old
        cdef cnp.ndarray[double, ndim=1] weights
        cdef double rho, z_j, l1_term
        cdef int iter, j

        # Convert weights to typed buffer
        weights = <cnp.ndarray[double, ndim=1]> self.weights

        # Initialize coefficients and residuals
        self.beta = np.zeros(n_features, dtype=np.float64)
        residual = y.copy()
        beta_old = np.zeros(n_features, dtype=np.float64)

        for iter in range(self.max_iter):
            # Copy beta for convergence check
            beta_old[:] = <cnp.ndarray[double, ndim=1]> self.beta

            for j in range(n_features):
                # Update residual excluding current feature
                residual += X[:, j] * self.beta[j]

                # Compute gradient and update coefficient
                rho = np.dot(X[:, j], residual) / n_samples
                z_j = np.dot(X[:, j], X[:, j]) / n_samples + self.alpha * (1 - self.l1_ratio) * weights[j]
                l1_term = self.alpha * self.l1_ratio * weights[j] # Weight both the L1 and L2 penalties by the feature weight

                if rho > l1_term:
                    self.beta[j] = (rho - l1_term) / z_j
                elif rho < -l1_term:
                    self.beta[j] = (rho + l1_term) / z_j
                else:
                    self.beta[j] = 0.0

                # Update residual
                residual -= X[:, j] * self.beta[j]

            # Check for convergence
            if np.max(np.abs(self.beta - beta_old)) < self.tol:
                break

    def predict(self, cnp.ndarray[double, ndim=2] X):
        """
        Predict using the trained model.

        Parameters:
            X (array): Feature matrix (n_samples x n_features).

        Returns:
            y_pred (array): Predicted values.
        """
        return np.dot(X, self.beta)

    def get_coefficients(self):
        """
        Get the fitted coefficients.

        Returns:
            beta (array): Coefficients of the model.
        """
        return self.beta
