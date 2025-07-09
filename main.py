import cv2
import numpy as np
from utils import extract_bit_plane, compute_psi, compute_weighted_histogram

cap = cv2.VideoCapture("images/templates/childs.mp4")
trackers = []

cv2.namedWindow("Frame")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        print("Select and press entry then esc")
        frozen_frame = frame.copy()
        rois = cv2.selectROIs("Selection ", frozen_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Selection")
        gray_frozen = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2GRAY)
        for roi in rois:
            x, y, w, h = roi
            roi_img = gray_frozen[y:y+h, x:x+w]
            T6 = extract_bit_plane(roi_img, 6)
            T7 = extract_bit_plane(roi_img, 7)
            hist = compute_weighted_histogram(roi_img)
            name = input(f"name the object : ")
            trackers.append({
                "name": name,
                "T6_base": T6,
                "T7_base": T7,
                "hist": hist,
                "w0": w,
                "h0": h,
                "last_x": x,
                "last_y": y
            })
        print(f"{len(trackers)} added")

    for obj in trackers:
        scales = [0.8, 1.0, 1.2]
        found = False
        best_score = -1
        best_coords = (obj["last_x"], obj["last_y"])
        best_size = (obj["w0"], obj["h0"])

        for scale in scales:
            w = int(obj["w0"] * scale)
            h = int(obj["h0"] * scale)
            T6 = cv2.resize(obj["T6_base"], (w, h), interpolation=cv2.INTER_NEAREST)
            T7 = cv2.resize(obj["T7_base"], (w, h), interpolation=cv2.INTER_NEAREST)

            x0 = max(0, obj["last_x"] - 30)
            x1 = min(gray.shape[1] - w, obj["last_x"] + 30)
            y0 = max(0, obj["last_y"] - 30)
            y1 = min(gray.shape[0] - h, obj["last_y"] + 30)

            for y in range(y0, y1, 2):
                for x in range(x0, x1, 2):
                    roi = gray[y:y + h, x:x + w]
                    I6 = extract_bit_plane(roi, 6)
                    I7 = extract_bit_plane(roi, 7)
                    zeros = compute_psi(T6, T7, I6, I7)
                    roi_hist = compute_weighted_histogram(roi)
                    hist_sim = cv2.compareHist(obj["hist"].astype(np.float32), roi_hist.astype(np.float32), cv2.HISTCMP_CORREL)
                    total_score = 0.8 * zeros + 0.2 * hist_sim

                    if total_score > best_score:
                        best_score = total_score
                        best_coords = (x, y)
                        best_size = (w, h)

        if best_score > 2750:
            found = True

        if not found:
            print(f"[{obj['name']}] not found locally, searching globally")
            for scale in scales:
                w = int(obj["w0"] * scale)
                h = int(obj["h0"] * scale)
                T6 = cv2.resize(obj["T6_base"], (w, h), interpolation=cv2.INTER_NEAREST)
                T7 = cv2.resize(obj["T7_base"], (w, h), interpolation=cv2.INTER_NEAREST)

                for y in range(0, gray.shape[0] - h, 4):
                    for x in range(0, gray.shape[1] - w, 4):
                        roi = gray[y:y + h, x:x + w]
                        I6 = extract_bit_plane(roi, 6)
                        I7 = extract_bit_plane(roi, 7)
                        zeros = compute_psi(T6, T7, I6, I7)
                        roi_hist = compute_weighted_histogram(roi)
                        hist_sim = cv2.compareHist(obj["hist"].astype(np.float32), roi_hist.astype(np.float32), cv2.HISTCMP_CORREL)
                        total_score = 0.8 * zeros + 0.2 * hist_sim

                        if total_score > best_score:
                            best_score = total_score
                            best_coords = (x, y)
                            best_size = (w, h)
                            found = True

        obj["last_x"], obj["last_y"] = best_coords
        x, y = best_coords
        w, h = best_size
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(display_frame, obj["name"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Frame", display_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()