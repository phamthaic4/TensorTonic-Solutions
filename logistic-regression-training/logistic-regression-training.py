import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, d = X.shape
    w = np.zeros(d)
    b = 0
    for ep in range(steps):
        A = _sigmoid(np.dot(X, w) + b)
        grad_w = np.dot(X.T, (A - y)) / N
        grad_b = np.sum(A - y) / N

        w = w - lr * grad_w
        b = b - lr * grad_b
    return (w, b)