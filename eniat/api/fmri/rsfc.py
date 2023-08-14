from typing import Optional, Union
import numpy as np
import sys
from ..stats import *
from ...helper import get_cluster_coordinates

def connectivity_strength(x: np.ndarray, y: Optional[np.ndarray] = None,
                          pval: Optional[float] = None,
                          pos=False, abs=False) -> np.ndarray:
    from ..stats.corr import corr, corr_with, r_to_t
    if y is None:
        r = corr(x)
        r[np.nonzero(np.eye(r.shape[0]))] = 0
    else:
        r = corr_with(x, y)

    if pos:
        r[r < 0] = 0
    if abs:
        r = np.abs(r)
    if pval is not None:
        t, p = r_to_t(r, x.shape[-1])
        r[p >= pval] = 0
    r[np.nonzero(np.eye(r.shape[0]))] = 0
    return r.sum(-1)


def regional_homogeneity(data, nn=3, io_handler=sys.stdout):
    from functools import partial
    mask_idx = np.nonzero(data.mean(-1))
    gcc = partial(get_cluster_coordinates, size=1, nn_level=nn)
    io_handler.write('Extracting nearest coordinates set...')
    all_coords_set = np.apply_along_axis(gcc, 0, np.array(mask_idx)).T
    io_handler.write('[Done]\n')
    masked_reho = np.zeros(all_coords_set.shape[0])

    n_voxels = all_coords_set.shape[0]
    progress = 1
    io_handler.write('Calculating regional homogeneity...\n')
    for i, coord in enumerate(all_coords_set):
        # filter outbound coordinate
        c_msk = []
        for j, arr in enumerate(coord):
            s = data.shape[j]
            c_msk.append(np.nonzero(arr > s - 1)[0])
        coord_flt = [f for f in range(coord.shape[-1]) if f not in list(set(np.concatenate(c_msk, 0)))]
        coord = coord[:, coord_flt]
        cd = data[tuple(coord)]
        masked_cd = cd[np.nonzero(cd.mean(-1))]
        masked_reho[i] = kendall_w(masked_cd)
        if (i / n_voxels) * 10 >= progress:
            io_handler.write(f'{progress}..')
            progress += 1
        if i == (n_voxels - 1):
            io_handler.write('10 [Done]\n')

    reho = np.zeros(data.shape[:3])
    reho[mask_idx] = masked_reho
    return reho


def amplitude_low_freq_fluctuation(data: np.ndarray,
                                   dt: Union[int, float],
                                   lowcut: Optional[float] = None, highcut: Optional[float] = None,
                                   pval: Optional[float]=None,
                                   fraction: bool=False,
                                   io_handler=sys.stdout):
    """ Amplitude of Low Frequency Fluctuation

    Args:
        data: V x T
        dt: sampling time
        lowcut: cut frequency for highpass filter
        highcut: cut frequency for lowpass filter

    Returns:
        ALFF
    """
    from ..signal import power_spectral_density
    io_handler.write('Calculating ALFF...')
    f, pxx = power_spectral_density(data, dt=dt)
    alff = pxx[..., (f >= lowcut) & (f <= highcut)].sum(-1)
    if fraction:
        alff[np.nonzero(pxx.sum(-1))] /= pxx.sum(-1)[np.nonzero(pxx.sum(-1))]
    io_handler.write('[Done]\n')
    return alff