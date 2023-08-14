from .io import decomp_dataobj
import numpy as np


def get_signal(func_obj, roi_obj):
    data, affine, _ = decomp_dataobj(func_obj)
    mask_idx = np.nonzero(data.mean(-1))
    roi, _, _ = decomp_dataobj(roi_obj)
    
    data_msk = data[mask_idx]
    roi_msk = roi[mask_idx]
    
    signal = data_msk * roi_msk[..., np.newaxis]
    return signal[np.nonzero(roi_msk)].mean(0)