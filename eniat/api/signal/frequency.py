from ...error import *
import numpy as np
from scipy import signal
from typing import Union, Optional, Tuple


def bandpass(data: np.ndarray,
             dt: float, order: int = 5,
             lowcut: Optional[float] = None,
             highcut: Optional[float] = None,
             output: str = 'sos',
             analog = False) -> np.ndarray:
    """
    Method to perform bandpass filtering. If only one frequency is given, perform Highpass filter instead.

    Args:
        data (np.ndarray): V x T data where V is voxels and T is time points.
        dt (Union[int, float]): Sampling time.
        order (int, optional): Order of the filter. Defaults to 5.
        lowcut (Optional[float], optional): Filter frequency cut for high-pass filter.
        highcut (Optional[float], optional): Filter frequency cut for low-pass filter.
        output (str, optional): Type of butter filter design. Defaults to 'sos'.
        analog (bool, optional): Analog filter design. Defaults to False.

    Returns:
        np.ndarray: Filtered signal.
        
    Raises:
        InvalidApproach: If both lowcut and highcut are not provided.
        InvalidApproach: If an invalid output type is provided.
        NotImplemented: If zpk method is not yet implemented.
    """
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if lowcut and highcut:
        op = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass', output=output, analog=analog)
    else:
        if lowcut:
            op = signal.butter(order, lowcut/nyq, btype='highpass', output=output, analog=analog)
        elif highcut:
            op = signal.butter(order, highcut/nyq, btype='lowpass', output=output, analog=analog)
        else:
            raise InvalidApproach('Missing filter frequency.')

    if output == 'sos':
        y = signal.sosfilt(op, data)
    elif output == 'ba':
        y = signal.lfilter(op[0], op[1], data)
    elif output == 'zpk':
        raise NotImplemented('Method not implemented yet.')
    else:
        raise InvalidApproach('Invalid metrics.')

    return y


def power_spectral_density(data: np.ndarray,
                           dt: Union[int, float],
                           nperseg: Optional[int] = None,
                           average: str = 'mean'
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method.

    Args:
        data (np.ndarray): V x T data where V is voxels and T is time points.
        dt (Union[int, float]): Sampling time.
        nperseg (Optional[int], optional): Length of each segment. Defaults to None.
        average (str, optional): Method to use when averaging periodograms. Defaults to 'mean'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of sample frequencies and power spectral density of data.
    """
    
    fs = 1.0 / dt
    input_length = data.shape[-1]
    if input_length < 256 and nperseg is None:
        nperseg = input_length

    f, pxx = signal.welch(data, fs=fs, window='hann', nperseg=nperseg,
                          scaling='density', average=average)
    return f, pxx


def phase_angle(data: np.ndarray) -> np.ndarray:
    analytic_signal = signal.hilbert(data)
    return np.rad2deg(np.angle(analytic_signal))