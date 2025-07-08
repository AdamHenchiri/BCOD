import cv2
import numpy as np
from utils import extract_bit_plane, compute_psi, compute_weighted_histogram

cap = cv2.VideoCapture("images/templates/childs.mp4")
object_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=80)

trackers = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for (bx, by, bw, bh) in param["boxes"]:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                roi = param["frame"][by:by + bh, bx:bx + bw]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                T6 = extract_bit_plane(gray_roi, 6)
                T7 = extract_bit_plane(gray_roi, 7)
                hist = compute_weighted_histogram(gray_roi)
                name = input(f"📝 Entrez un nom pour l'objet sélectionné ({bx}, {by}): ")
                trackers.append({
                    "name": name,
                    "T6_base": T6,
                    "T7_base": T7,
                    "hist": hist,
                    "w0": bw,
                    "h0": bh,
                    "last_x": bx,
                    "last_y": by
                })
                print(f"{name} is selected to be tracked!")
                break

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", on_mouse, param={})

while True:
    #lecture de la video
    ret, frame = cap.read()
    if not ret:
        break
    #Pour faire la rotation
    #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # application du mask pour récupérer les objets en mouvements
    mask = object_detector.apply(frame)
    # prendre que les objet en pur blanc
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.setMouseCallback("Frame", on_mouse, param={"frame": frame.copy(), "boxes": boxes})

    for obj in trackers:
        scales = [0.8, 1.0, 1.2]
        found = False
        best_score = -1
        best_coords = (obj["last_x"], obj["last_y"])
        best_size = (obj["w0"], obj["h0"])

        # Recherche locale a 30px de rayon
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
                    #boolean compt
                    zeros = compute_psi(T6, T7, I6, I7)
                    #histogram compute
                    roi_hist = compute_weighted_histogram(roi)
                    hist_sim = cv2.compareHist(obj["hist"].astype(np.float32), roi_hist.astype(np.float32),
                                               cv2.HISTCMP_CORREL)
                    total_score = 0.8 * zeros + 0.2 * hist_sim
                    # print("zero score = ",zeros)
                    # print("hist_sim score = ",hist_sim)
                    # print("total score = ",total_score)
                    if total_score > best_score:
                        best_score = total_score
                        best_coords = (x, y)
                        best_size = (w, h)

                    print(best_score)
                    if best_score > 2750:
                        found = True
                    else:
                        found = False

        # Recherche globale si objet non trouvé localement
        if not found:
            print('not found !')
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
                        hist_sim = cv2.compareHist(obj["hist"].astype(np.float32), roi_hist.astype(np.float32),
                                                   cv2.HISTCMP_CORREL)
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
