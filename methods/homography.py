import cv2
import numpy as np
from methods.base import TrackerMethod

class HomographyTracker(TrackerMethod):
    def __init__(self, init_frame, roi):
        x, y, w, h = map(int, roi)
        self.template = cv2.cvtColor(init_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(1000)
        self.kp_template, self.des_template = self.orb.detectAndCompute(self.template, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)
        if des_frame is None or self.des_template is None:
            return None, 0

        matches = self.bf.knnMatch(self.des_template, des_frame, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) < 4:
            return None, 0

        pts_template = np.float32([self.kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, 5.0)
        if H is None:
            return None, 0

        projected = cv2.perspectiveTransform(self.corners, H)
        x, y, w, h = cv2.boundingRect(projected)
        return (x, y, w, h), len(good)
