# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:11:41 2017

@author: subum
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:28:15 2017

@author: subum
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:52:23 2017

@author: subum
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import scipy.ndimage



def gs_filter(img, sigma):
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, sigma)


def gradient_intensity(img):
    preSobel = np.array([[[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                         [[4, 8, 4], [0, 0, 0], [-4, -8, -4]],
                         [[2, 4, 2], [0, 0, 0], [-2, -4, -2]]], np.int32)
    # Kernel for Gradient in y-direction
    dx = ndimage.sobel(preSobel, 0)  # x derivative
    dy = ndimage.sobel(preSobel, 1)  # y derivative
    dz = ndimage.sobel(preSobel, 2)  # z derivative
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, dx)
    Iy = ndimage.filters.convolve(img, dy)
    Iz = ndimage.filters.convolve(img, dz)

    G = abs(Ix) + abs(Iy) + abs(Iz)

    D2 = np.arctan2(Iz, Iy)
    print(D2)
    return (G, D2)


def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle


def suppression(img, D):
    M, N, O = img.shape
    Z = np.zeros((M, N, O), dtype=np.int32)

    for i in range(M):
        for j in range(N):
            for k in range(O):
                # find neighbour pixels to visit from the gradient directions
                where = round_angle(D[i, j, k])
                try:
                    if where == 0:
                        if (img[i, j, k] >= img[i, j - 1, k]) and (img[i, j, k] >= img[i, j + 1, k]):
                            Z[i, j, k] = img[i, j, k]
                    elif where == 90:
                        if ((img[i, j, k] >= img[i - 1, j, k]) and (img[i, j, k] >= img[i + 1, j, k])
                            and (img[i, j, k] >= img[i, j, k - 1]) and (img[i, j, k] >= img[i, j, k + 1])):
                            Z[i, j, k] = img[i, j, k]
                    elif where == 135:
                        if ((img[i, j, k] >= img[i, j + 1, k + 1]) and (img[i, j, k] >= img[i, j - 1, k + 1])
                            and (img[i, j, k] >= img[i + 1, j, k + 1]) and (img[i, j, k] >= img[i - 1, j, k + 1])
                            and (img[i, j, k] >= img[i, j + 1, k - 1]) and (img[i, j, k] >= img[i, j - 1, k - 1])
                            and (img[i, j, k] >= img[i + 1, j, k - 1]) and (img[i, j, k] >= img[i - 1, j, k - 1])):
                            Z[i, j, k] = img[i, j, k]
                    elif where == 45:
                        if ((img[i, j, k] >= img[i + 1, j + 1, k + 1]) and (img[i, j, k] >= img[i + 1, j - 1, k - 1])
                            and (img[i, j, k] >= img[i + 1, j - 1, k + 1]) and (img[i, j, k] >= img[i - 1, j + 1, k + 1])
                            and (img[i, j, k] >= img[i - 1, j - 1, k + 1]) and (img[i, j, k] >= img[i + 1, j + 1, k - 1])
                            and (img[i, j, k] >= img[i - 1, j + 1, k - 1]) and (img[i, j, k] >= img[i + 1, j, k - 1])
                            and (img[i, j, k] >= img[i - 1, j - 1, k]) and (img[i, j, k] >= img[i + 1, j + 1, k])
                            and (img[i, j, k] >= img[i - 1, j + 1, k]) and (img[i, j, k] >= img[i + 1, j - 1, k])):
                            Z[i, j, k] = img[i, j, k]
                except IndexError as e:
                    """ Todo: Deal with pixels at the image boundaries. """
                    pass
    return Z


def threshold(img, t, T):
    cf = {'WEAK': np.int32(50), 'STRONG': np.int32(255), }

    # get strong pixel indices
    strong_i, strong_j, strong_k = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j, weak_k = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j, zero_k = np.where(img < t)

    # set values
    img[strong_i, strong_j, strong_k] = cf.get('STRONG')
    img[weak_i, weak_j, weak_k] = cf.get('WEAK')
    img[zero_i, zero_j, zero_k] = np.int32(0)

    return (img, cf.get('WEAK'))


def tracking(img, weak, strong=255):
    M, N, O = img.shape
    for i in range(M):
        for j in range(N):
            for k in range(O):
                if img[i, j, k] == weak:
                    # check if one of the neighbours is strong (=255 by default)
                    try:
                        if ((img[i, j + 1, k] == strong) or (img[i, j + 1, k + 1] == strong)
                            or (img[i, j + 1, k - 1] == strong) or (img[i, j - 1, k] == strong)
                            or (img[i, j - 1, k + 1] == strong) or (img[i, j - 1, k - 1] == strong)
                            or (img[i + 1, j, k] == strong) or (img[i, j, k + 1] == strong)
                            or (img[i, j, k - 1] == strong) or (img[i - 1, j, k] == strong)
                            or (img[i - 1, j, k + 1] == strong) or (img[i - 1, j, k - 1] == strong)
                            or (img[i - 1, j - 1, k] == strong) or (img[i - 1, j, k + 1] == strong)
                            or (img[i - 1, j - 1, k - 1] == strong) or (img[i - 1, j + 1, k] == strong)
                            or (img[i - 1, j + 1, k + 1] == strong) or (img[i - 1, j + 1, k - 1] == strong)
                            or (img[i + 1, j, k + 1] == strong) or (img[i + 1, j, k - 1] == strong)
                            or (img[i + 1, j + 1, k] == strong) or (img[i + 1, j + 1, k + 1] == strong)
                            or (img[i + 1, j + 1, k - 1] == strong) or (img[i + 1, j - 1, k] == strong)
                            or (img[i + 1, j - 1, k - 1] == strong) or (img[i + 1, j - 1, k + 1] == strong)):
                            img[i, j, k] = strong
                        else:
                            img[i, j, k] = 0
                    except IndexError as e:
                        pass
    return img


