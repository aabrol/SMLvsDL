import nibabel as nib
import numpy as np

class MRI:
    def __init__(self, series_uid):
        mri_path = f'/home/feczk001/shared/projects/S1067_Loes/experiments/scans/{series_uid}.nii.gz'
        img = nib.load(mri_path)
        self.struct_arr = np.array(img.get_data(), dtype=np.float32)
