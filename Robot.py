import cv2
import numpy as np

class Robot:
    def __init__(self, qr_data, points):
        self.qr_data = qr_data
        self.points = points
        self.last_seen = 0
        self.kalman = self.init_kalman_filter(points)
        self.predicted_center = self.compute_center(points)

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

    def update(self, points, frame_num):
        self.points = points
        self.last_seen = frame_num
        center = self.compute_center(points)
        measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])  # Shape (2,1)
        self.kalman.correct(measurement)
        self.predicted_center = center

    def predict(self):
        prediction = self.kalman.predict()
        self.predicted_center = (float(prediction[0]), float(prediction[1]))
        return self.predicted_center

    def is_active(self, frame_num, max_lost=30):
        return (frame_num - self.last_seen) < max_lost
