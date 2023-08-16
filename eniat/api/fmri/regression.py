import numpy as np
from typing import Union, Optional, IO
import nibabel as nib

class DualReg:
    """
    Dual regression integration using QR decomposition
    """
    def __init__(self, data, model):
        """
        data: VxT matrix where V is the number of voxels, T is the number of timepoints
        model: VxF matrix where V is the number of voxels, F is the number of spatial features
        """
        # private
        self._data = data
        self._model = model
        self._Tt, self._Pt = None, None
        self._Ts, self._Ps = None, None
        self._Coefs, self._Coeft = None, None
        
    def fit(self, pval=None):
        """
        fit data to model
        """
        self._dual_regression(pval)
        
    def _dual_regression(self, pval=None):
        """
        Perform spatial regression followed by the temporal regression using QR decomposition
        """
        from ..regression import linear_regression
        from scipy import stats
        model = self._model
        
        predicted_t, self._Coeft = linear_regression(self._data, model, method='svd', return_beta=True)
        v, f = model.shape
        dof = v - f
        mse_t = np.square(predicted_t.sum(-1) - self._data).sum(-1) / float(dof)
        diag_t = np.linalg.inv(np.dot(model.T, model)).diagonal()
        se_t = np.sqrt((mse_t * np.concatenate([diag_t[:, np.newaxis]], axis=-1)).T)
        
        t_t = self._Coeft.copy()
        t_t[se_t == 0] = 0
        t_t[np.nonzero(se_t)] /= se_t[np.nonzero(se_t)]
        
        self._Tt = t_t
        self._Pt = 2 * (1 - stats.t.cdf(abs(t_t), df=dof))
        
        model = self._Coeft
        predicted_s, self._Coefs = linear_regression(self._data.T, model, method='svd', return_beta=True)
        v, f = model.shape
        dof = v - f
        mse_s = np.square(predicted_s.sum(-1) - self._data.T).sum(-1) / float(dof)
        diag_s = np.linalg.inv(np.dot(model.T, model)).diagonal()
        se_s = np.sqrt((mse_s * np.concatenate([diag_s[:, np.newaxis]], axis=-1)).T)
        
        t_s = self._Coefs.copy()
        t_s[se_s == 0] = 0
        t_s[np.nonzero(se_s)] /= se_s[np.nonzero(se_s)]
        
        self._Ts = t_s
        self._Ps = 2 * (1 - stats.t.cdf(abs(t_s), df=dof))
        if pval is not None:
            self._Coeft[self._Pt > pval] = 0
            self._Coefs[self._Ps > pval] = 0
    
    @property
    def coef_(self):
        """
        Coefficients for each spatial and temporal features
        """
        return dict(spatial=self._Coefs,  # VxF matrix
                    temporal=self._Coeft) # FxT matrix
    
    @property
    def predicted(self):
        """
        Predicted data from dual regression

        Return:
            np.ndarray (2D): VxT matrix
        """
        return np.dot(self._Coefs, self._Coeft)
    
    @property
    def decoded(self):
        """
        Decoded data using model
        
        Return:
            np.ndarray (2D): VxT matrix
        """
        return np.dot(self._model, self._Coeft)

    @property
    def error(self):
        """
        The sum of square of difference between original data and predicted data
        
        Return:
            np.ndarray (1D)
        """
        return np.square(self._data - self.predicted).sum(0)
    

def nuisance_regression(nib_img, mask: Optional[nib.Nifti1Image] = None,
                        port: int =1,
                        ort: Optional[np.ndarray] = None,
                        dt: Union[int, float] = 1.0,
                        lowcut: Optional[float] = 0.01,
                        highcut: Optional[float] = 0.1,
                        fwhm: Optional[float] = None,
                        verbose: bool = True,
                        io_handler: Optional[IO] = None):
    from ...helper import decomp_dataobj, save_to_nib, sitk2nib, nib2sitk
    from ..regression import polynomial_feature, linear_regression
    from ..signal import bandpass
    from ..ndimage import gaussian_smoothing


    if io_handler is None:
        if verbose:
            from sys import stdout
            io_handler = stdout
        else:
            from ...helper import SilentIO
            io_handler = SilentIO()
    data, affine, resol = decomp_dataobj(nib_img)
    if mask is None:
        mask_idx = np.nonzero(data.std(-1))
    else:
        mask_idx = np.nonzero(mask.dataobj)
    data_masked = data[mask_idx]

    # detrend data
    model = polynomial_feature(data_masked, order=port)
    model /= model.max(axis=0, initial=None)
    if ort is not None:
        ort -= ort.mean(0)
        ort /= abs(ort).max(0)
        model = np.concatenate([model, ort], axis=-1)
    fit_masked = linear_regression(data_masked, model=model)
    data_masked -= fit_masked.sum(-1)

    # bandpass filter
    data_masked = bandpass(data_masked, dt=dt, lowcut=lowcut, highcut=highcut)
    data_masked += fit_masked[..., 0]
    data[mask_idx] = data_masked
    filtered_nii = save_to_nib(data, affine)

    # gaussian smoothing
    if fwhm is not None:
        sitk_img, header = nib2sitk(filtered_nii)
        filtered_sitk_img = gaussian_smoothing(sitk_img, fwhm, io_handler=io_handler)
        filtered_nii = sitk2nib(filtered_sitk_img, header)
    return filtered_nii