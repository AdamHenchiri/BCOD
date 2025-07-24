import cv2
import numpy as np
from utils import extract_bit_plane, compute_psi, compute_weighted_histogram

class Robot:
    def __init__(self, qr_data, points, frame_roi=None):
        self.qr_data = qr_data
        self.points = points
        self.last_seen = 0
        self.kalman = self.init_kalman_filter(points)
        self.predicted_center = self.compute_center(points)

        # Bit plane tracking attributes
        self.T6_base = None
        self.T7_base = None
        self.hist = None
        self.w0 = 0
        self.h0 = 0
        self.last_x = 0
        self.last_y = 0
        self.tracking_mode = "qr"
        self.qr_lost_frames = 0

        # Initialize bit plane tracking if we have a ROI
        if frame_roi is not None:
            self.init_bitplane_tracking(frame_roi)

    def init_bitplane_tracking(self, frame_roi):
        """Initialize bit plane tracking templates from QR region"""
        x, y, w, h = cv2.boundingRect(self.points.astype(int))

        # Expand bounding box slightly for better tracking
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame_roi.shape[1] - x, w + 2 * margin)
        h = min(frame_roi.shape[0] - y, h + 2 * margin)

        roi = frame_roi[y:y + h, x:x + w]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            self.T6_base = extract_bit_plane(gray_roi, 6)
            self.T7_base = extract_bit_plane(gray_roi, 7)
            self.hist = compute_weighted_histogram(gray_roi)
            self.w0 = w
            self.h0 = h
            self.last_x = x
            self.last_y = y

    def compute_center(self, pts):
        return np.mean(pts, axis=0)

    def init_kalman_filter(self, points):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], dtype=np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        center = self.compute_center(points)
        kalman.statePre = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
        kalman.statePost = kalman.statePre.copy()

        return kalman

    def track_bitplane(self, gray_frame, frame_num):
        """Track using bit plane method when QR is lost"""
        if self.T6_base is None or self.T7_base is None:
            return False

        self.qr_lost_frames += 1
        self.tracking_mode = "bitplane"

        scales = [0.8, 1.0, 1.2]
        found = False
        best_score = -1
        best_coords = (self.last_x, self.last_y)
        best_size = (self.w0, self.h0)

        # Local search first (30px radius)
        search_radius = 30
        for scale in scales:
            w = int(self.w0 * scale)
            h = int(self.h0 * scale)

            if w <= 0 or h <= 0:
                continue

            T6 = cv2.resize(self.T6_base, (w, h), interpolation=cv2.INTER_NEAREST)
            T7 = cv2.resize(self.T7_base, (w, h), interpolation=cv2.INTER_NEAREST)

            x0 = max(0, self.last_x - search_radius)
            x1 = min(gray_frame.shape[1] - w, self.last_x + search_radius)
            y0 = max(0, self.last_y - search_radius)
            y1 = min(gray_frame.shape[0] - h, self.last_y + search_radius)

            for y in range(y0, y1, 2):
                for x in range(x0, x1, 2):
                    if x + w > gray_frame.shape[1] or y + h > gray_frame.shape[0]:
                        continue

                    roi = gray_frame[y:y + h, x:x + w]
                    if roi.size == 0:
                        continue

                    I6 = extract_bit_plane(roi, 6)
                    I7 = extract_bit_plane(roi, 7)
                    zeros = compute_psi(T6, T7, I6, I7)

                    roi_hist = compute_weighted_histogram(roi)
                    hist_sim = cv2.compareHist(self.hist.astype(np.float32),
                                               roi_hist.astype(np.float32),
                                               cv2.HISTCMP_CORREL)

                    total_score = 0.8 * zeros + 0.2 * hist_sim

                    if total_score > best_score:
                        best_score = total_score
                        best_coords = (x, y)
                        best_size = (w, h)

        # Threshold for successful tracking
        if best_score > 2750:
            found = True

        # Global search if local search failed
        if not found and self.qr_lost_frames < 60:  # Limit global search
            for scale in scales:
                w = int(self.w0 * scale)
                h = int(self.h0 * scale)

                if w <= 0 or h <= 0:
                    continue

                T6 = cv2.resize(self.T6_base, (w, h), interpolation=cv2.INTER_NEAREST)
                T7 = cv2.resize(self.T7_base, (w, h), interpolation=cv2.INTER_NEAREST)

                for y in range(0, gray_frame.shape[0] - h, 8):  # Increased step for speed
                    for x in range(0, gray_frame.shape[1] - w, 8):
                        roi = gray_frame[y:y + h, x:x + w]
                        if roi.size == 0:
                            continue

                        I6 = extract_bit_plane(roi, 6)
                        I7 = extract_bit_plane(roi, 7)
                        zeros = compute_psi(T6, T7, I6, I7)

                        roi_hist = compute_weighted_histogram(roi)
                        hist_sim = cv2.compareHist(self.hist.astype(np.float32),
                                                   roi_hist.astype(np.float32),
                                                   cv2.HISTCMP_CORREL)

                        total_score = 0.8 * zeros + 0.2 * hist_sim

                        if total_score > best_score:
                            best_score = total_score
                            best_coords = (x, y)
                            best_size = (w, h)
                            if total_score > 2750:
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

        # Update position and Kalman filter
        if found:
            self.last_x, self.last_y = best_coords
            self.w0, self.h0 = best_size

            # Update Kalman filter with bit plane tracking result
            center_x = best_coords[0] + best_size[0] / 2
            center_y = best_coords[1] + best_size[1] / 2
            measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
            self.kalman.correct(measurement)
            self.predicted_center = (center_x, center_y)
            self.last_seen = frame_num

        return found

    def is_active(self, frame_num, max_lost=90):  # Increased timeout for bit plane tracking
        return (frame_num - self.last_seen) < max_lost

    def update_qr(self, points, frame_num):
        """Update with QR detection"""
        self.points = points
        self.last_seen = frame_num
        self.qr_lost_frames = 0
        self.tracking_mode = "qr"

        center = self.compute_center(points)
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
        self.kalman.correct(measurement)
        self.predicted_center = center

        # Update bit plane tracking position
        self.last_x = int(center[0] - self.w0 / 2)
        self.last_y = int(center[1] - self.h0 / 2)

    def predict(self):
        prediction = self.kalman.predict()
        self.predicted_center = (float(prediction[0]), float(prediction[1]))
        return self.predicted_center

    def get_bounding_box(self):
        """Get current bounding box for visualization"""
        if self.tracking_mode == "qr":
            return cv2.boundingRect(self.points.astype(int))
        else:
            return (self.last_x, self.last_y, self.w0, self.h0)
