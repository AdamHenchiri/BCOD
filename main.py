import cv2
import numpy as np
from utils import extract_bit_plane, compute_psi, compute_weighted_histogram
from pyzbar.pyzbar import decode

def detect_qr_codes(frame, gray_frame):
    global trackers, detected_qr_codes

    decoded_objects = decode(frame)
    qr_codes = []

    for obj in decoded_objects:
        qr_data = obj.data.decode("utf-8")
        points = obj.polygon

        if len(points) >= 4:
            pts = np.array([(p.x, p.y) for p in points], dtype=np.int32)
            x = np.min(pts[:, 0])
            y = np.min(pts[:, 1])
            w = np.max(pts[:, 0]) - x
            h = np.max(pts[:, 1]) - y
        else:
            continue

        qr_id = f"{qr_data}_{x}_{y}"

        already_tracking = False
        for tracker in trackers:
            if tracker["qr_data"] == qr_data:
                already_tracking = True
                break
        if not already_tracking and qr_id not in detected_qr_codes:
            roi = gray_frame[y:y + h, x:x + w]
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                T6 = extract_bit_plane(roi, 6)
                T7 = extract_bit_plane(roi, 7)
                hist = compute_weighted_histogram(roi)

                trackers.append({
                    "name": f"QR_{len(trackers)}",
                    "qr_data": qr_data,
                    "qr_type": "QRCODE",
                    "T6_base": T6,
                    "T7_base": T7,
                    "hist": hist,
                    "w0": w,
                    "h0": h,
                    "last_x": x,
                    "last_y": y,
                    "last_angle": 0,
                    "qr_points": pts,
                    "recent_detection": True
                })

                detected_qr_codes.add(qr_id)
                print(f"[NEW QR] {qr_data} at ({x}, {y})")

        qr_codes.append({
            'data': qr_data,
            'points': pts,
            'rect': (x, y, w, h)
        })

    return qr_codes


# cap = cv2.VideoCapture("images/templates/qr_code2.mp4")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

trackers = []
detected_qr_codes = {}

cv2.namedWindow("Frame")


def rotate_image(image, angle):
    """Optimized rotation function"""
    if angle == 0:
        return image

    h, w = image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR)


def compute_matching_score(T6, T7, hist_template, roi):
    """Compute matching score between template and ROI"""
    I6 = extract_bit_plane(roi, 6)
    I7 = extract_bit_plane(roi, 7)
    zeros = compute_psi(T6, T7, I6, I7)
    roi_hist = compute_weighted_histogram(roi)
    hist_sim = cv2.compareHist(hist_template.astype(np.float32),
                               roi_hist.astype(np.float32),
                               cv2.HISTCMP_CORREL)
    return 0.8 * zeros + 0.2 * hist_sim


def get_rotated_rectangle_points(center, size, angle):
    """Get the 4 corner points of a rotated rectangle"""
    w, h = size
    angle_rad = np.radians(angle)

    # Half dimensions
    hw, hh = w / 2.0, h / 2.0

    # Define corners relative to center
    corners = np.array([
        [-hw, -hh],  # Top-left
        [hw, -hh],  # Top-right
        [hw, hh],  # Bottom-right
        [-hw, hh]  # Bottom-left
    ])

    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    # Apply rotation
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Translate to actual center position
    rotated_corners[:, 0] += center[0]
    rotated_corners[:, 1] += center[1]

    return rotated_corners.astype(np.int32)

def track_object_locally(obj, gray):
    """Local tracking in search area"""
    best_score = -1
    best_coords = (obj["last_x"], obj["last_y"])
    best_angle = obj.get("last_angle", 0)
    found = False

    w, h = obj["w0"], obj["h0"]
    T6_base = obj["T6_base"]
    T7_base = obj["T7_base"]

    # Define search area
    search_radius = 50
    search_box = [
        max(0, obj["last_x"] - search_radius),
        min(gray.shape[1] - w, obj["last_x"] + search_radius),
        max(0, obj["last_y"] - search_radius),
        min(gray.shape[0] - h, obj["last_y"] + search_radius)
    ]

    # Search with step for optimization
    step = 16
    for y in range(search_box[2], search_box[3], step):
        for x in range(search_box[0], search_box[1], step):
            if x + w > gray.shape[1] or y + h > gray.shape[0]:
                continue

            roi = gray[y:y + h, x:x + w]
            if roi.shape[0] != h or roi.shape[1] != w:
                continue

            score = compute_matching_score(T6_base, T7_base, obj["hist"], roi)

            if score > best_score:
                best_score = score
                best_coords = (x, y)
                best_angle = 0
                found = True

    return found, best_score, best_coords, best_angle


def track_object_globally(obj, gray):
    """Global tracking with rotation"""
    best_score = -1
    best_coords = (obj["last_x"], obj["last_y"])
    best_angle = obj.get("last_angle", 0)
    found = False

    w, h = obj["w0"], obj["h0"]
    T6_base = obj["T6_base"]
    T7_base = obj["T7_base"]

    # Test different angles around last known angle
    last_angle = obj.get("last_angle", 0)
    angle_step = 5
    angle_range = range(last_angle - 45, last_angle + 46, angle_step)

    for angle in angle_range:
        angle = angle % 360

        # Rotate template
        if angle == 0:
            rot_T6, rot_T7 = T6_base, T7_base
        else:
            rot_T6 = rotate_image(T6_base, angle)
            rot_T7 = rotate_image(T7_base, angle)

        rh, rw = rot_T6.shape

        # Search with larger step for speed
        step = 8
        for y in range(0, gray.shape[0] - rh, step):
            for x in range(0, gray.shape[1] - rw, step):
                roi = gray[y:y + rh, x:x + rw]
                if roi.shape[0] != rh or roi.shape[1] != rw:
                    continue

                score = compute_matching_score(rot_T6, rot_T7, obj["hist"], roi)

                if score > best_score:
                    best_score = score
                    best_coords = (x, y)
                    best_angle = angle
                    found = True

    return found, best_score, best_coords, best_angle


print("Instructions:")
print("- Press 'q' to quit")
print("- Press 'p' to manually select regions (optional)")
print("- Press 'd' to toggle QR code detection on/off")
print("- QR codes will be automatically detected and tracked")

auto_detect = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(30) & 0xFF
    decoded_objects = decode(frame)

    to_delete = []
    for k, v in detected_qr_codes.items():
        v["ttl"] -= 1
        if v["ttl"] <= 0:
            to_delete.append(k)
    for k in to_delete:
        del detected_qr_codes[k]

    for obj in decoded_objects:
        qr_data = obj.data.decode("utf-8")
        points = obj.polygon
        if len(points) >= 4:
            pts = np.array([(p.x, p.y) for p in points], dtype=np.int32)

            # Affichage du contour
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            cv2.putText(frame, qr_data[:20], (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Mémorisation simple avec TTL
            detected_qr_codes[qr_data] = {
                "points": pts,
                "ttl": 30  # persiste 30 frames sans redétection
            }

    # Affichage optionnel des QR non vus récemment (gris)
    for qr_data, info in detected_qr_codes.items():
        if info["ttl"] < 30:
            cv2.polylines(frame, [info["points"]], True, (150, 150, 150), 1)

    cv2.imshow("QR Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()