import numpy as np
import nibabel as nib

class SilentIO:
    """
    A file-like object that suppresses all write operations. 
    Useful for redirecting and silencing output.
    
    Examples:
    --------
    ```python
    import sys
    sys.stdout = SilentIO()  # Redirect standard output to suppress print statements
    print("This won't be printed to the console.")
    ```
    """
    def __init__(self):
        """Initializes a new instance of the SilentIO class."""
        pass

    def write(self, message: str) -> None:
        """
        Overridden write method that does nothing.
        
        Parameters:
        - message : str
            The message intended to be written. In this implementation, the message will be discarded.
        """
        pass


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