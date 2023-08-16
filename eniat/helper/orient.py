import numpy as np
from .io import decomp_dataobj, save_to_nib
from scipy.ndimage import affine_transform


def reorient_to_ras(data, affine, resol):
    """ Reorient and re-sample the input data into RAS space
    to ensure consistent orientation regardless of which axis
    was sliced during data acquisition.

    Returns:
        ras_data
        ras_resol
    """
    np.set_printoptions(precision=4, suppress=True)
    shape = data.shape[:3]
    fov_size = np.asarray(shape) * resol
    rotate_mat = np.round(affine[:3, :3] / resol, decimals=0).astype(np.int8)
    origin = rotate_mat.dot(fov_size/2)

    org_affine = affine.copy()
    org_affine[:3, 3] = -origin

    ras_resol = abs(rotate_mat.dot(resol))
    ras_shape = abs(rotate_mat.dot(shape))
    ras_affine = np.eye(4)
    ras_affine[:3, :3] = np.diag(resol)
    ras_affine[:3, 3] = -fov_size/2

    org_mm2vox = np.linalg.inv(org_affine)
    ras_mm2vox = org_mm2vox.dot(ras_affine)

    rotate = ras_mm2vox[:3, :3]
    shift = ras_mm2vox[:3, 3]

    if len(data.shape) == 4:
        ras_data = np.zeros(data.shape)
        for f in range(data.shape[-1]):
            ras_data[:,:,:, f] = affine_transform(data[:,:,:, f],
                                                  rotate, shift,
                                                  output_shape=ras_shape)
    else:
        ras_data = affine_transform(data, rotate, shift,
                                    output_shape=ras_shape)
    return ras_data, ras_resol


def determine_slice_plane(slice_axis, affine, resol):
    """ return the original scheme of slice plane """
    rotate_mat = (affine[:3, :3] / resol).astype(np.int8)
    ras_axis = abs(rotate_mat.dot(range(3))).tolist()
    return ['sagittal', 'coronal', 'axial'][ras_axis.index(slice_axis)]


def from_matvec(mat, vec):
    affine = np.eye(4)
    affine[:3,:3] = mat
    affine[:3, 3] = vec
    return affine


def to_matvec(matrix):
    return matrix[:3, :3], matrix[:3, 3]


def norm_orient(nib_img):
    """ Reorient and re-sample the input data into RAS space
    to ensure consistent orientation regardless of which axis
    was sliced during data acquisition.

    Returns:
        nib.NifTi1Image obj
    """
    data, affine, resol = decomp_dataobj(nib_img)

    np.set_printoptions(precision=4, suppress=True)
    shape = data.shape
    fov_size = np.asarray(shape) * np.array(resol)
    rotate_mat = np.round(affine[:3, :3] / np.array(resol)).astype(int)

    origin = rotate_mat.dot(fov_size / 2)
    org_affine = affine.copy()
    org_affine[:3, 3] = -origin

    ras_resol = abs(rotate_mat.dot(resol))
    ras_shape = abs(rotate_mat.dot(shape))
    ras_affine = np.eye(4)
    ras_affine[:3, :3] = np.diag(ras_resol * np.array([1, -1, 1]))
    ras_affine[:3, 3] = -fov_size / 2 * np.array([1, -1, 1])

    org_mm2vox = np.linalg.inv(org_affine)
    ras_mm2vox = org_mm2vox.dot(ras_affine)

    rotate = ras_mm2vox[:3, :3]
    shift = ras_mm2vox[:3, 3]

    ras_data = affine_transform(data, rotate, shift, output_shape=ras_shape)
    ras_affine[:3, :3] = ras_affine[:3, :3]
    ras_affine[:3, 3] = rotate_mat.dot(affine[:3, 3]) * np.array([1, -1, 1])

    return save_to_nib(ras_data, ras_affine)


def get_coord_center_of_mass(func_obj):
    """ Return center of mass coordinate on Paxinos system """
    from scipy import ndimage
    
    mask = np.array(func_obj.dataobj).astype(bool)
    mask_com = ndimage.measurements.center_of_mass(mask)
    affine = func_obj.affine.copy()
    com_coord = affine[:3, :3].dot(mask_com) + affine[:3, 3]
    return com_coord * np.array([1, 1, -1]) - np.array([0, 0.36, -7])


def get_slice(niiobj, coord):
    data, affine, resol = decomp_dataobj(niiobj)
    data = np.round(data, decimals=10)
    
    # prepare coordinate system
    x, y, z = get_meshgrid(niiobj)
    
    axi_img = data[:, :, coord[2]].astype(float)
    cor_img = data[:, coord[1], :].T.astype(float)
    sag_img = data[coord[0], :, :].T.astype(float)
    
    axi_img[axi_img == 0] = np.nan
    cor_img[cor_img == 0] = np.nan
    sag_img[sag_img == 0] = np.nan
    
    return dict(axial=(y, x, axi_img),
                coronal=(x, z, cor_img),
                sagittal=(y, z, sag_img))


def get_meshgrid(niiobj):
    data, affine, resol = decomp_dataobj(niiobj)
    size = data.shape
    x0, y0, z0 = affine[:3, :3].dot(np.array([0, 0, 0])) + affine[:3, 3]
    x1, y1, z1 = affine[:3, :3].dot(np.array(size[:3])) + affine[:3, 3]
    
    x = np.linspace(x0, x1, size[0])
    y = np.linspace(y0, y1, size[1])
    z = np.linspace(z0, z1, size[2])
    return x, y, z


def paxinose_to_camri(x, y, z):
    return np.array([x, y, z]) * np.array([1, 1, -1]) + np.array([0, 0.36, 7])

def correct_affine(nib_obj):
    # method to correct affine
    # temporary method for v2-1 template
    data, affine, _ = decomp_dataobj(nib_obj)
    affine[0, 3] -= 0.2
    return save_to_nib(data, affine)

def mm_to_voxel(coord, affine):
    new_coord = np.linalg.inv(affine[:3, :3]).dot(np.array(coord)-affine[:3, 3])
    return np.round(new_coord, decimals=0).astype(int)