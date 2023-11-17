import numpy as np
import nibabel
from scipy import ndimage

def resize(data, target_W=256, target_D=256, target_H=256):
    data = data.get_fdata()
    [width, depth, height] = data.shape
    scale = [target_W*1.0/width, target_D*1.0/depth, target_H*1.0/height]
    data = ndimage.zoom(data, scale, order=0)
    data = data.astype(np.float32)

    return data

if __name__ == '__main__':
    import sys
    # Load data from args
    data_path = sys.argv[1]
    data = nibabel.load(data_path)
    affine = data.affine
    # Resize data
    data = resize(data)
    # Save data
    nibabel.save(nibabel.Nifti1Image(data, affine), data_path+'resized.nii.gz')
