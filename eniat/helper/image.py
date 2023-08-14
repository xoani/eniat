import numpy as np
from typing import Tuple


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
