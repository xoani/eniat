import os
import re
import numpy as np
import nibabel as nib
from .image import random_rgb


class Atlas:
    """ This class templating the segmentation image object to handle atlas related attributes
    """
    def __init__(self, fname, dir='./', img_ext='nii.gz', label_ext='label'):
        # privates
        self._labels = dict()
        self._cmap = dict()
        self._conf = dict()

        if fname.endswith(img_ext):
            self.img_path = os.path.join(dir, f'{fname}')
        else:
            self.img_path = os.path.join(dir, f'{fname}.{img_ext}')
        self.label_path = os.path.join(dir, f'{fname}.{label_ext}')
        self._load_img()
        self._load_label()

    def _load_img(self):
        self.imgobj = nib.load(self.img_path)

    def _load_label(self):
        if os.path.exists(self.label_path):
            pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
                      r'(?P<Opac>(\d+|\d+\.\d+))\s+(?P<Vis2d>\d+)\s+(?P<Vis3d>\d+)\s+"(?P<roi>.*)$'
            with open(self.label_path, 'r') as label_f:
                for line in label_f:
                    if re.match(pattern, line):
                        idx = int(re.sub(pattern, r'\g<idx>', line))
                        self._labels[idx] = re.sub(pattern, r'\g<roi>', line).split('"')[0]
                        rgb = re.sub(pattern, r"\g<R> \g<G> \g<B>", line)
                        rgb = rgb.split()
                        self._cmap[idx] = np.array([c for c in map(float, rgb)]) / 255
                        con = re.sub(pattern, r"\g<Opac> \g<Vis2d> \g<Vis3d>", line).split()
                        self._conf[idx] = dict(Opacity=float(con[0]), Vis2d=int(con[1]), Vis3d=int(con[2]))
        else:
            labels = set(np.asarray(self.imgobj.dataobj).astype(int).flatten())
            num_zfill = len(str(max(labels)))
            for idx in labels:
                if idx == 0:
                    label = "Clear Label"
                    cmap = np.array([0, 0, 0])
                    conf = dict(Opacity=0, Vis2d=0, Vis3d=0)
                else:
                    label = f"Label {str(idx).zfill(num_zfill)}"
                    cmap = np.array([c for c in map(float, random_rgb())]) / 255
                    conf = dict(Opacity=1, Vis2d=1, Vis3d=0)
                self._labels[idx] = label
                self._cmap[idx] = cmap
                self._conf[idx] = conf

    def save_label(self, fname, dir='./', label_ext='label'):
        """
        Save label instant to new file
        Args:
            fname: file name
            dir: parent dir
            label_ext: file extension
        """
        new_label_path = os.path.join(dir, f'{fname}.{label_ext}')

        with open(new_label_path, 'w') as f:
            line = list()
            for idx in self.labels.keys():
                roi = self.labels[idx]
                rgb = self.cmap[idx]
                rgb = np.array(rgb) * 255
                rgb = rgb.astype(int)
                con = self._conf[idx]
                if idx == 0:
                    line = '{:>5}   {:>3}  {:>3}  {:>3}        '\
                           '0  0  0    "{}"\n'.format(idx, rgb[0], rgb[1], rgb[2], roi)
                else:
                    line = '{}{:>5}   {:>3}  {:>3}  {:>3}        '\
                           '{}  {}  {}    "{}"\n'.format(line, idx, rgb[0], rgb[1], rgb[2],
                                                         con['Opacity'], con['Vis2d'], con['Vis3d'],
                                                         roi)
            f.write(line)

    @property
    def labels(self):
        return self._labels

    @property
    def cmap(self):
        return self._cmap

    def __getitem__(self, idx: int):
        if idx != 0:
            selected_roi = np.zeros(self.imgobj.shape)
            selected_roi[np.asarray(self.imgobj.dataobj) == idx] = 1
            roi_imgobj = nib.Nifti1Image(selected_roi, affine=self.imgobj.affine, header=self.imgobj.header)
            return self.labels[idx], roi_imgobj
        else:
            return None

    def __repr__(self):
        labels = None
        for idx in self._labels.keys():
            if not idx:
                labels = '[{:>3}] {:>40}\n'.format(idx, self.labels[idx])
            else:
                labels = '{}[{:>3}] {:>40}\n'.format(labels, idx, self.labels[idx])
        return labels