from typing import Union, Optional
from pathlib import Path
import nibabel as nib

class BaseBrainViewer:
    def __init__(self, 
                 underlay_image_path: Optional[Union[str, Path]] = None, 
                 overlay_image_path: Optional[Union[str, Path]] = None):
        if underlay_image_path:
            self.set_underlay_image(underlay_image_path)
        if overlay_image_path:
            self.set_overlay_image(overlay_image_path)

    def set_underlay_image(self, image_path: Union[str, Path]):
        self.underlay_image = nib.load(image_path)

    def set_overlay_image(self, image_path: Union[str, Path]):
        self.overlay_image = nib.load(image_path)


class BrainViewer2D(BaseBrainViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
