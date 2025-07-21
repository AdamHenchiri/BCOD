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

    if retval and points is not None:
        for i, qr_data in enumerate(decoded_info):
            if not qr_data:
                continue

            pts = np.int32(points[i]).reshape(-1, 2)
            robots.setdefault(qr_data, Robot(qr_data, pts, frame_id)).update(pts, frame_id)

    # Nettoyage des robots inactifs
    inactive_qr = [qr for qr, rob in robots.items() if not rob.is_active(frame_id)]
    for qr in inactive_qr:
        del robots[qr]

    # Affichage
    for qr_data, rob in robots.items():
        cv2.polylines(display, [np.int32(rob.points)], True, (0, 255, 0), 2)
        x, y = rob.points[0]
        cv2.putText(display, qr_data[:20], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("QR Tracker", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
