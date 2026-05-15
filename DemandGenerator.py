# Import required dependencies
import numpy as np

def random_recursive_signal(n_samples: np.int32, start: np.float64, scale: np.float64=1.0) -> np.ndarray:
    """Build a random-walk signal by recursively adding Gaussian noise."""
    signal = np.zeros(shape=n_samples, dtype=np.float64)

    # Each noise sample changes the next point relative to the previous point.
    noise = np.random.normal(loc=0.0, scale=scale, size=n_samples - 1)

    # Set the starting point without noise
    signal[0] = start

    for i in range(1, n_samples):
        signal[i] = signal[i - 1] + noise[i - 1]

    return signal

def scale_signal(signal: np.ndarray, method: str='MinMax') -> np.ndarray:
    """Scale a signal with either MinMax normalization or standardization."""
    # Lowercase the selector so callers do not need to match exact casing.
    method_lower_case = method.lower()

    match method_lower_case:
        case 'minmax':
            _min, _max = np.min(a=signal), np.max(a=signal)
            if np.isclose(_max, _min):
                return np.zeros_like(signal, dtype=np.float64)
            return (signal - _min) / (_max - _min)

        case 'std':
            mu, sigma = np.mean(a=signal), np.std(a=signal)
            if np.isclose(sigma, 0.0):
                return np.zeros_like(signal, dtype=np.float64)
            return (signal - mu) / sigma

        case _:
            raise ValueError(f"Error: '{method}' method is not recognised")
        
def moving_average_filter(signal: np.ndarray, window_size: np.int32=7) -> np.ndarray:
    """Smooth a signal with a moving-average filter."""
    if window_size <= 0:
        raise ValueError("Error: window size must be greater than zero!")
    if window_size > signal.shape[0]:
        raise ValueError("Error: window size must not exceed signal length!")
    
    # Pad the end so every original sample has a full averaging window.
    p_size         = window_size - 1

    # Repeat the last value instead of padding with zeros, which would create an
    # artificial downward tail in the demand signal.
    signal_padding                   = np.zeros(shape=signal.shape[0] + p_size, dtype=np.float64)
    signal_padding[:signal.shape[0]] = signal
    signal_padding[signal.shape[0]:] = signal[-1]

    output_signal = np.zeros(shape=signal.shape[0], dtype=np.float64)
    for i in range(signal.shape[0]):
        output_signal[i] = np.mean(a=signal_padding[i : i + window_size])

    return output_signal

def generate_demand(n_samples: np.int32, start: np.float64=None, scale: np.float64=None, apply_filtering: bool=True) -> np.ndarray:
    """Generate a normalized random power-demand time series."""
    # Use random defaults when the caller does not provide a starting point or
    # random-walk volatility.
    demand_signal = random_recursive_signal(n_samples=n_samples,
                                            start=start if start is not None else np.random.uniform(low=0.0, high=100.0),
                                            scale=scale if scale is not None else 1.0)
    
    # Convert the demand to the same normalized 0..1 scale used by the MDP.
    demand_signal_norm = scale_signal(signal=demand_signal, method='MinMax')

    if apply_filtering:
        # Remove short-term noise so the simulated demand changes more smoothly.
        return moving_average_filter(signal=demand_signal_norm)
    
    return demand_signal_norm
