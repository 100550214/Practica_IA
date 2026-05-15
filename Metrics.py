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
    return 1 - (numerator/denominator)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    numerator = np.float64(np.cov(y_true,y_pred))
    denominator = np.float64(np.std(y_true)*np.std(y_pred))
    return numerator/denominator
