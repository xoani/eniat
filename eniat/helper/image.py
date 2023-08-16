import numpy as np
from typing import Tuple
from .io import decomp_dataobj, save_to_nib


def random_rgb() -> Tuple[int, int, int]:
    """
    Generate a random RGB color using a predefined set of intensity levels.
    
    Returns:
    - tuple
        A tuple of three integers representing a random RGB color.
    """
    levels = range(32,256,32)
    return tuple(np.random.choice(levels) for _ in range(3))


def get_rgb_tuple(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB channels from integer format (0-255) to float format (0-1).
    
    Parameters:
    - r, g, b : int
        RGB channels as integers.
        
    Returns:
    - tuple
        A tuple of three floats representing the RGB color in [0, 1] range.
    """
    return r / 255., g / 255., b / 255.


def get_cluster_coordinates(coord: Tuple[int, int, int], size: int=1, nn_level: int=3) -> Tuple[int, int, int]:
    """
    Generate a list of 3D coordinates representing a cluster around the provided center.
    
    Parameters:
    - coord : Tuple[int, int, int]
        The center of the cluster.
    - size : int, optional (default=1)
        The size of the cluster from the center.
    - nn_level : int, optional (default=3)
        Defines the type of neighbors to include: 
        1 = faces only, 
        2 = faces and edges, 
        3 = faces, edges, and corners.

    Returns:
    - list of tuples
        List of 3D coordinates representing the cluster.
    """
    n_voxel = size + 1
    x, y, z = coord
    x_ = sorted([x + i for i in range(n_voxel)] + [x - i for i in range(n_voxel) if i != 0])
    y_ = sorted([y + i for i in range(n_voxel)] + [y - i for i in range(n_voxel) if i != 0])
    z_ = sorted([z + i for i in range(n_voxel)] + [z - i for i in range(n_voxel) if i != 0])

    if nn_level == 1:
        thr = size
    elif nn_level == 2:
        thr = np.sqrt(np.square([size] * 2).sum())
    elif nn_level == 3:
        thr = np.sqrt(np.square([size] * 3).sum())
    else:
        raise ValueError('[nn_level] only accept a value in [1, 2, 3]')

    all_poss = [(i, j, k) for i in x_ for j in y_ for k in z_]
    output_coord = [c for c in all_poss if cal_distance(coord, c) <= thr]

    return output_coord


def cal_distance(coord_a: Tuple[float, float, float], coord_b: Tuple[float, float, float]) -> float:
    """
    Calculate the Euclidean distance between two 3D coordinates.
    
    Parameters:
    - coord_a, coord_b : Tuple[float, float, float]
        The two 3D coordinates.

    Returns:
    - float
        The Euclidean distance between the two coordinates.
    """
    return np.sqrt(np.square(np.diff(np.asarray(list(zip(coord_a, coord_b))))).sum())


def fwhm2sigma(fwhm: float) -> float:
    """
    Convert Full Width at Half Maximum (FWHM) to standard deviation (sigma) 
    for a Gaussian distribution.

    Parameters:
    - fwhm (float): The Full Width at Half Maximum value.

    Returns:
    - float: The standard deviation (sigma) of the Gaussian distribution corresponding to the provided FWHM.
    """
    return fwhm / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma: float) -> float:
    """
    Convert standard deviation (sigma) of a Gaussian distribution 
    to its Full Width at Half Maximum (FWHM).

    Parameters:
    - sigma (float): The standard deviation (sigma) of the Gaussian distribution.

    Returns:
    - float: The Full Width at Half Maximum (FWHM) corresponding to the provided sigma.
    """
    return sigma * np.sqrt(8 * np.log(2))


def pad_by_voxel(nib_img, a=None, p=None, l=None, r=None, i=None, s=None):
    """
    a: anterior
    p: posterior
    l: left
    r: right
    s: superior
    i: inferior
    """
    options = dict(a=a, p=p, l=l, r=r, i=i, s=s)
    data, affine, resol = decomp_dataobj(nib_img)
    x, y, z = data.shape
    for k, o in options.items():
        if o is None:
            pass
        else:
            if k == 'a':
                affine[1, 3] += resol[1] * o
                pad = np.zeros([x, o, z])
                data = np.concatenate([pad, data], axis=1)
                x, y, z = data.shape
            if k == 'p':
                pad = np.zeros([x, o, z])
                data = np.concatenate([data, pad], axis=1)
                x, y, z = data.shape
            if k == 'l':
                affine[0, 3] += resol[0] * o
                pad = np.zeros([o, y, z])
                data = np.concatenate([pad, data], axis=0)
                x, y, z = data.shape
            if k == 'r':
                pad = np.zeros([o, y, z])
                data = np.concatenate([data, pad], axis=0)
                x, y, z = data.shape
            if k == 'i':
                affine[2, 3] -= resol[2] * o
                pad = np.zeros([x, y, o])
                data = np.concatenate([pad, data], axis=2)
                x, y, z = data.shape
            if k == 's':
                pad = np.zeros([x, y, o])
                data = np.concatenate([data, pad], axis=2)
                x, y, z = data.shape
    return save_to_nib(data, affine)


def crop_by_voxel(nib_img, a=None, p=None, l=None, r=None, i=None, s=None):
    """
    a: anterior
    p: posterior
    l: left
    r: right
    s: superior
    i: inferior
    """
    options = dict(a=a, p=p, l=l, r=r, i=i, s=s)
    data, affine, resol = decomp_dataobj(nib_img)
    x, y, z = data.shape
    for k, o in options.items():
        if o is None:
            pass
        else:
            if k == 'a':
                affine[1, 3] += resol[1] * o
                data = data[:, o:, :]
                x, y, z = data.shape
            if k == 'p':
                data = data[:, :(y-o), :]
                x, y, z = data.shape
            if k == 'l':
                affine[0, 3] += resol[0] * o
                data = data[o:, :, :]
                x, y, z = data.shape
            if k == 'r':
                data = data[:(x-o), :, :]
                x, y, z = data.shape
            if k == 'i':
                affine[2, 3] += resol[2] * o
                data = data[:, :, o:]
                x, y, z = data.shape
            if k == 's':
                pdata = data[:, :, :(z-o)]
                x, y, z = data.shape
    return save_to_nib(data, affine)


def concat_3d_to_4d(*nib_objs):
    concat_data = []
    affine = None
    for nii in nib_objs:
        data, affine, _ = decomp_dataobj(nii)
        concat_data.append(data[..., np.newaxis])
    concat_data = np.concatenate(concat_data, axis=-1)
    return save_to_nib(concat_data, affine)


def blur(data, sigma):
    from scipy.ndimage import gaussian_filter
    
    data_mask = (data == 0)
    data = gaussian_filter(data, sigma).astype(float)
    data[data_mask] = np.nan
    return data