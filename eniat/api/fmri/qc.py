import sys
from typing import Union
import numpy as np


def tsnr(data: np.ndarray, io_handler=sys.stdout) -> Union[np.ndarray, int]:
    """ calculate temporal snr

    Args:
        data: 1d or 2d (V x T) data

    Returns:
        tsnr
    """
    io_handler.write('Calculating tSNR...')
    dim = data.shape
    if dim == 1:
        tsnr = data.mean() / data.std()
    else:
        mean = data.mean(-1)
        std = data.std(-1)
        tsnr = np.zeros(mean.shape)
        masked_mean = mean[np.nonzero(std)]
        masked_std = std[np.nonzero(std)]
        tsnr[np.nonzero(std)] = masked_mean / masked_std
    io_handler.write('[Done]\n')
    return tsnr