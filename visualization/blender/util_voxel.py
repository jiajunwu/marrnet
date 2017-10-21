import numpy as np

def pooling(mat, step, method):
    assert(len(mat.shape) == 3)
    assert(mat.shape[0] % step == 0 and mat.shape[1] % step == 0 and mat.shape[2] % step == 0)
    if method == 'mean':
        mat = mat.reshape(int(mat.shape[0]/step), step, mat.shape[1], mat.shape[2]).mean(1).squeeze()
        mat = mat.reshape(mat.shape[0], int(mat.shape[1]/step), step, mat.shape[2]).mean(2).squeeze()
        mat = mat.reshape(mat.shape[0], mat.shape[1], int(mat.shape[2]/step), step).mean(3).squeeze()
    elif method == 'max':
        mat = mat.reshape(int(mat.shape[0]/step), step, mat.shape[1], mat.shape[2]).max(1).squeeze()
        mat = mat.reshape(mat.shape[0], int(mat.shape[1]/step), step, mat.shape[2]).max(2).squeeze()
        mat = mat.reshape(mat.shape[0], mat.shape[1], int(mat.shape[2]/step), step).max(3).squeeze()
    else:
        error('Unknown pooling method. Only mean and max are supported')
    return mat
