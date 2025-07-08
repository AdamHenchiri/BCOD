import cv2
import numpy as np

def extract_bit_plane(image, bit):
    return ((image >> bit) & 1).astype(np.uint8)

def compute_psi(T6, T7, I6, I7):
    xor7 = cv2.bitwise_xor(T7, I7)
    xor6 = cv2.bitwise_xor(T6, I6)
    or_result = cv2.bitwise_or(xor7, xor6)
    zeros = np.count_nonzero(or_result == 0)
    return zeros

def compute_weighted_histogram(gray_roi):
    """
    to apply a gaussien filter on the object detected
    :param gray_roi:
    :return:
    """
    hist = cv2.calcHist([gray_roi], [0], None, [32], [0, 256])
    return hist.flatten() / (np.sum(hist) + 1e-6)