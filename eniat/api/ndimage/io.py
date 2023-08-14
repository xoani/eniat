import numpy as np
import nibabel as nib

def decomp_dataobj(nib_img):
    data = np.asarray(nib_img.dataobj).copy()
    affine = nib_img.affine.copy()
    resol = nib_img.header['pixdim'][1:4]
    return data, affine, resol

def save_to_nib(data, affine):
    nii = nib.Nifti1Image(data, affine)
    nii.header['sform_code'] = 0
    nii.header['qform_code'] = 1
    return nii