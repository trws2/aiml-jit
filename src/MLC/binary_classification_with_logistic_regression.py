import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    tmp = np.dot(X, weights) + bias
    p = 1 / (1 + np.exp(-tmp))
    tmp = p >= 0.5 
    tmp = tmp.astype(int)
    return tmp.tolist()

print(predict_logistic(np.array([[0, 0], [0.1, 0.1], [-0.1, -0.1]]), np.array([1, 1]), 0))

