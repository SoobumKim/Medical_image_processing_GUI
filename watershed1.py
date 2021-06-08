import cv2
from scipy.ndimage import label
import numpy
from skimage import color

def water(image):
    thresh,b1 = cv2.threshold(image, 0, 255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    # since Otsu's method has over segmented the image
    # erosion operation is performed
    b2 = cv2.erode(b1, None,iterations = 3)
    # distance transform is performed
    dist_trans = cv2.distanceTransform(b2, 2, 3)
    # thresholding the distance transform image to obtain
    # pixels that are foreground
    thresh, dt = cv2.threshold(dist_trans, 1, 255, cv2.THRESH_BINARY)
    # performing labeling
    #labelled = label(b, background = 0)
    labelled, ncc = label(dt)
    # labelled is converted to 32-bit integer
    labelled = labelled.astype(numpy.int32)
    a1 = color.gray2rgb(image)
    return(a1, labelled)

