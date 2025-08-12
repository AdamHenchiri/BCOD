import cv2
import numpy as np
from methods.base import TrackerMethod

class ORBTracker(TrackerMethod):
    def __init__(self, init_frame, roi):
        x, y, w, h = map(int, roi)
        self.template = cv2.cvtColor(init_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(500)
        self.kp_template, self.des_template = self.orb.detectAndCompute(self.template, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.w, self.h = w, h

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray, None)
        if des_frame is None or self.des_template is None:
            return None, 0
        matches = self.bf.match(self.des_template, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            return None, 0

        pts_template = np.float32([self.kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_template, pts_frame, cv2.RANSAC, 5.0)
        if H is None:
            return None, 0

        corners = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)
        x, y, w, h = cv2.boundingRect(projected)
        return (x, y, w, h), len(matches)
