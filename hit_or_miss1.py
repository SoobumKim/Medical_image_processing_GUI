import scipy.misc
import numpy as np
import scipy.ndimage
from skimage import filters

def hit_miss(image):

    thresh = filters.threshold_otsu(image)
    image = image > thresh

    structure1 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [1, 0, 0]]])

    structure2 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 1]]])

    structure3 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 0, 0], [0, 0, 0], [0, 0, 0]]])

    structure4 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 1], [0, 0, 0], [0, 0, 0]]])

    structure5 = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    structure6 = np.array([[[0, 0, 1], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    structure7 = np.array([[[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    structure8 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    # performing the binary hit-or-miss
    b = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure1)
    # b is converted from an ndarray to an image
    b1 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure2)

    b2 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure3)

    b3 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure4)

    b4 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure5)
    # b is converted from an ndarray to an image
    b5 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure6)

    b6 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure7)

    b7 = scipy.ndimage.morphology.binary_hit_or_miss(image, structure1=structure8)

    k = b|b1|b2|b3|b4|b5|b6|b7
    return(k)