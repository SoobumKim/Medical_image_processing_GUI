import numpy as np
from skimage.filters import threshold_otsu
import skimage.exposure as imexp


def renyi_seg_fn(im, alpha):

    hist, bins = np.histogram(im, 256, [0, 255])
    # Convert all values to float
    hist_float = [float(i) for i in hist]
    # compute the pdf
    pdf = hist_float / np.sum(hist_float)
    # compute the cdf
    cumsum_pdf = np.cumsum(pdf)
    s = 0
    e = 255  # assuming 8 bit image
    scalar = 1.0 / (1 - alpha)
    # A very small value to prevent
    # division by zero
    eps = np.spacing(1)

    rr = e - s
    # The second parentheses is needed because
    # the parameters are tuple
    h1 = np.zeros((rr, 1))
    h2 = np.zeros((rr, 1))
    # the following loop computes h1 and h2
    # values used to compute the entropy
    for ii in range(1, rr):
        iidash = ii + s

        temp1 = np.power((pdf[0:iidash] / cumsum_pdf[iidash]), alpha)
        h1[ii] = scalar * np.log2(np.sum(temp1) + eps)
        temp2 = np.power((pdf[iidash + 1:256] / (1 - cumsum_pdf[iidash])), alpha)
        h2[ii] = scalar * np.log2(np.sum(temp2) + eps)

    T = h1 + h2
    # Entropy value is calculated

    # location where the maximum entropy
    # occurs is the threshold for the renyi entropy
    location = T.argmax(axis=0)
    # location value is used as the threshold
    thresh = location
    return thresh
