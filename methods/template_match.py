import cv2
from methods.base import TrackerMethod

class TemplateMatchTracker(TrackerMethod):
    def __init__(self, init_frame, roi, method=cv2.TM_CCOEFF_NORMED):
        x, y, w, h = map(int, roi)
        self.template = cv2.cvtColor(init_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        self.w, self.h = w, h
        self.method = method

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            score = -min_val
        else:
            top_left = max_loc
            score = max_val
        bbox = (top_left[0], top_left[1], self.w, self.h)
        return bbox, float(score)
