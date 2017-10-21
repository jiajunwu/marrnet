# this file is to be called from shell, not within blender

import os
import sys
import numpy as np
from scipy.io import loadmat

filename = sys.argv[1]
iname = 'voxels'
if len(sys.argv) > 2:
	iname = sys.argv[2]
assert(filename[-4:] == '.mat')
mat = loadmat(filename)[iname].astype('double')

if len(mat.shape) == 4:
	mat = np.resize(mat, (mat.shape[0], 1, mat.shape[1], mat.shape[2], mat.shape[3]))
mat = mat[:, :, ::-1, :, :]


# write to file
np.save(filename[:-4] + '.npy', mat)

