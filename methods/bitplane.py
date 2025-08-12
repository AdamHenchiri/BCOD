import cv2
import numpy as np
from methods.base import TrackerMethod
from utils import extract_bit_plane_numba, compute_psi_numba


class BitplaneTracker(TrackerMethod):
    def __init__(self, init_frame, roi, score_threshold=2500, search_radius=50):
        x, y, w, h = map(int, roi)
        self.last_x, self.last_y = x, y
        self.w, self.h = w, h
        self.score_threshold = score_threshold
        self.search_radius = search_radius
        self.confidence = 1.0  # augmente si trouvé, baisse si perdu

        gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y:y + h, x:x + w]

        # Pré-calcul des bitplanes du template
        self.template_b6 = extract_bit_plane_numba(self.template, 6)
        self.template_b7 = extract_bit_plane_numba(self.template, 7)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # Définir la zone de recherche locale
        r = int(self.search_radius * (2.0 - self.confidence))
        x0, x1 = max(0, self.last_x - r), min(W - self.w, self.last_x + r)
        y0, y1 = max(0, self.last_y - r), min(H - self.h, self.last_y + r)

        best_score = -1
        best_bbox = (self.last_x, self.last_y, self.w, self.h)

        # Stride adaptatif
        stride = max(1, min(4, int(4 * (1.5 - self.confidence))))

        for yy in range(y0, y1, stride):
            for xx in range(x0, x1, stride):
                roi = gray[yy:yy + self.h, xx:xx + self.w]
                if roi.shape[0] != self.h or roi.shape[1] != self.w:
                    continue

                I6 = extract_bit_plane_numba(roi, 6)
                I7 = extract_bit_plane_numba(roi, 7)
                score = compute_psi_numba(self.template_b6, self.template_b7, I6, I7)

                if score > best_score:
                    best_score = score
                    best_bbox = (xx, yy, self.w, self.h)

        # Met à jour la position et la confiance
        self.last_x, self.last_y = best_bbox[0], best_bbox[1]
        if best_score > self.score_threshold:
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.confidence = max(0.1, self.confidence - 0.2)

        return best_bbox, float(best_score)
