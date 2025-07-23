import cv2
import numpy as np
import time
from Robot import Robot

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

qr_detector = cv2.QRCodeDetector()
robots = {}  # key = qr_data, value = Robot object
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    display = frame.copy()

    # Prétraitement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)

    seen_now = set()
    if retval:
        for i, data in enumerate(decoded_info):
            if not data: continue
            pts = np.int32(points[i]).reshape(-1, 2)
            seen_now.add(data)

            if data not in robots:
                robots[data] = Robot(data, pts)
            robots[data].update(pts, frame_id)

    for qr_id, robot in robots.items():
        if qr_id in seen_now:
            cv2.polylines(frame, [np.int32(robot.points)], True, (0, 255, 0), 2)
        elif robot.is_active(frame_id):
            pred = robot.predict()
            cv2.circle(frame, (int(pred[0]), int(pred[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{qr_id} (pred)", (int(pred[0]), int(pred[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Nettoyage
    to_delete = [r for r in robots if not robots[r].is_active(frame_id)]
    for r in to_delete:
        del robots[r]

    cv2.imshow("QR Tracking with Kalman", frame)
    frame_id += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()