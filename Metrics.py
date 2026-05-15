# Import required dependencies
import numpy as np

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    return np.float64(np.mean(np.abs(y_true-y_pred)))

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    return np.float64(np.mean(np.square(y_true - y_pred)))

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the R2 metric """
    numerator = np.float64(np.sum(np.square(y_true - y_pred)))
    denominator = np.float64(np.sum(np.square(y_true - np.mean(y_true))))
    if np.isclose(denominator, 0.0):
        return np.float64(1.0 if np.isclose(numerator, 0.0) else 0.0)
    return 1 - (numerator/denominator)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return np.float64(0.0)
    return np.float64(np.corrcoef(y_true, y_pred)[0, 1])
