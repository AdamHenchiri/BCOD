import cv2
import numpy as np
from numba import jit


def extract_bit_plane(image, bit):
    return ((image >> bit) & 1).astype(np.uint8)


@jit(nopython=True)
def extract_bit_plane_numba(image, bit):
    return ((image >> bit) & 1).astype(np.uint8)


def compute_psi(T6, T7, I6, I7):
    xor7 = cv2.bitwise_xor(T7, I7)
    xor6 = cv2.bitwise_xor(T6, I6)
    or_result = cv2.bitwise_or(xor7, xor6)
    zeros = np.count_nonzero(or_result == 0)
    return zeros


@jit(nopython=True)
def compute_psi_numba(T6, T7, I6, I7):
    """Numba-optimized PSI computation with early termination"""
    h, w = T6.shape
    zeros = 0
    total_pixels = h * w
    threshold = int(0.7 * total_pixels)  # Early termination threshold

    for i in range(h):
        for j in range(w):
            xor7 = T7[i, j] ^ I7[i, j]
            xor6 = T6[i, j] ^ I6[i, j]
            if (xor7 | xor6) == 0:
                zeros += 1

        # Early termination if we can't reach minimum threshold
        remaining_pixels = (h - i - 1) * w
        if zeros + remaining_pixels < threshold:
            return zeros

    return zeros


def compute_weighted_histogram(gray_roi):
    """
    to apply a gaussien filter on the object detected
    """
    hist = cv2.calcHist([gray_roi], [0], None, [32], [0, 256])
    return hist.flatten() / (np.sum(hist) + 1e-6)


def compute_weighted_histogram_optimized(gray_roi):
    """Optimized histogram computation with float32 output"""
    if gray_roi.size == 0:
        return np.zeros(32, dtype=np.float32)

    hist = cv2.calcHist([gray_roi], [0], None, [32], [0, 256])
    hist_sum = np.sum(hist)
    return (hist.flatten() / (hist_sum + 1e-6)).astype(np.float32)


