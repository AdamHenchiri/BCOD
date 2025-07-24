import cv2
import numpy as np
import time
from Robot import Robot


def main():
    cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     raise IOError("Cannot open webcam")

    qr_detector = cv2.QRCodeDetector()
    robots = {}  # key = qr_data, value = Robot object
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        display = frame.copy()

        # Preprocessing for QR detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhanced preprocessing for better QR detection
        gray_processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_processed = clahe.apply(gray_processed)
        gray_processed = cv2.GaussianBlur(gray_processed, (5, 5), 0)

        # QR Detection
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray_processed)

        seen_now = set()
        if retval:
            for i, data in enumerate(decoded_info):
                if not data:
                    continue
                pts = np.int32(points[i]).reshape(-1, 2)
                seen_now.add(data)

                if data not in robots:
                    # Create new robot with bit plane initialization
                    robots[data] = Robot(data, pts, frame)
                    print(f"New robot detected: {data}")
                else:
                    # Update existing robot with QR detection
                    robots[data].update_qr(pts, frame_id)

        # Track robots that lost QR detection using bit plane method
        for qr_id, robot in robots.items():
            if qr_id not in seen_now:
                # Try bit plane tracking
                found = robot.track_bitplane(gray, frame_id)
                if not found and robot.qr_lost_frames > 30:
                    # Use Kalman prediction as fallback
                    robot.predict()

        # Visualization
        for qr_id, robot in robots.items():
            if qr_id in seen_now:
                # QR detected - draw green polygon
                cv2.polylines(display, [np.int32(robot.points)], True, (0, 255, 0), 2)
                cv2.putText(display, f"{qr_id} (QR)",
                            tuple(map(int, robot.predicted_center)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif robot.is_active(frame_id):
                if robot.tracking_mode == "bitplane":
                    # Bit plane tracking - draw blue rectangle
                    x, y, w, h = robot.get_bounding_box()
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(display, f"{qr_id} (BP)", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                else:
                    # Kalman prediction - draw red circle
                    pred = robot.predict()
                    cv2.circle(display, (int(pred[0]), int(pred[1])), 8, (0, 0, 255), -1)
                    cv2.putText(display, f"{qr_id} (pred)",
                                (int(pred[0]), int(pred[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Cleanup inactive robots
        to_delete = [r for r in robots if not robots[r].is_active(frame_id)]
        for r in to_delete:
            print(f"Removing inactive robot: {r}")
            del robots[r]

        # Display tracking info
        info_text = f"Frame: {frame_id} | Robots: {len(robots)}"
        cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for i, (qr_id, robot) in enumerate(robots.items()):
            mode_text = f"{qr_id}: {robot.tracking_mode}"
            cv2.putText(display, mode_text, (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Hybrid QR + Bit Plane Tracking", display)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()