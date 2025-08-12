import cv2
import numpy as np
from utils import extract_bit_plane, compute_psi, compute_weighted_histogram, extract_bit_plane_numba, \
    compute_weighted_histogram_optimized, compute_psi_numba
from concurrent.futures import ThreadPoolExecutor, as_completed


class OptimizedTracker:
    def __init__(self):
        self.trackers = []
        self.search_radius = 30
        self.scales = [0.8, 1.0, 1.2]
        self.score_threshold = 2000
        self.angle_history_len = 5

    def _normalize_angle(self, angle):
        """Normalise angle"""
        return angle % 360

    def add_tracker(self, name, roi, center_x, center_y, w, h, angle=0):
        """Add a new tracker with RBB support and original template preservation"""
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        # Validation de la ROI
        if gray_roi.size == 0 or w < 10 or h < 10:
            print(f"ROI too small {name}")
            return False

        templates = {}
        for scale in self.scales:
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            if scaled_w > 0 and scaled_h > 0:
                scaled_roi = cv2.resize(gray_roi, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                templates[scale] = {
                    'T6': extract_bit_plane_numba(scaled_roi, 6),
                    'T7': extract_bit_plane_numba(scaled_roi, 7),
                    'size': (scaled_w, scaled_h)
                }

        if not templates:
            print(f"cannot create this template {name}")
            return False

        original_hist = compute_weighted_histogram_optimized(gray_roi)
        normalized_initial_angle = self._normalize_angle(angle)

        tracker = {
            "name": name,
            "templates": templates,
            "original_hist": original_hist.copy(),
            "current_hist": original_hist.copy(),
            "w0": w,
            "h0": h,
            "center_x": center_x,
            "center_y": center_y,
            "angle": normalized_initial_angle,
            "angle_history": [normalized_initial_angle] * self.angle_history_len,
            "confidence": 1.0,
            "lost_frames": 0,
            "update_count": 0,
            "drift_protection": True,
            "original_score_threshold": 0.3
        }
        self.trackers.append(tracker)
        return True

    def get_rotated_roi(self, gray, center_x, center_y, w, h, angle):
        """Extrait une ROI avec rotation de manière robuste"""
        # S'assurer que les dimensions sont positives
        if w <= 0 or h <= 0:
            # print(f"Debug: get_rotated_roi: Dimensions invalides w={w}, h={h}")
            return None, None

        # Calculer les 4 coins du rectangle non rotaté centré à (0,0)
        rect_pts_local = np.array([
            [-w / 2, -h / 2],
            [w / 2, -h / 2],
            [w / 2, h / 2],
            [-w / 2, h / 2]
        ], dtype=np.float32)

        # Appliquer la rotation à ces points autour de (0,0)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_mat_2x2 = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)
        rotated_local_pts = rect_pts_local @ rot_mat_2x2.T

        # Trouver la boîte englobante (AABB) de ces points rotatés
        min_x_rot_local, min_y_rot_local = np.min(rotated_local_pts, axis=0)
        max_x_rot_local, max_y_rot_local = np.max(rotated_local_pts, axis=0)

        bbox_w_large = int(np.ceil(max_x_rot_local - min_x_rot_local))
        bbox_h_large = int(np.ceil(max_y_rot_local - min_y_rot_local))

        if bbox_w_large <= 0 or bbox_h_large <= 0:
            # print(f"Debug: get_rotated_roi: Bbox calculée dégénérée w={bbox_w_large}, h={bbox_h_large}")
            return None, None

        # Déterminer la position de cette AABB dans l'image originale
        top_left_x_orig = int(center_x + min_x_rot_local)
        top_left_y_orig = int(center_y + min_y_rot_local)

        # Limiter cette AABB aux bords de l'image
        clip_x0 = max(0, top_left_x_orig)
        clip_y0 = max(0, top_left_y_orig)
        clip_x1 = min(gray.shape[1], top_left_x_orig + bbox_w_large)
        clip_y1 = min(gray.shape[0], top_left_y_orig + bbox_h_large)

        # Extraire la ROI "large" de l'image originale
        if clip_x1 <= clip_x0 or clip_y1 <= clip_y0:
            # print("Debug: get_rotated_roi: Bbox clippée est vide ou invalide.")
            return None, None

        roi_large = gray[clip_y0:clip_y1, clip_x0:clip_x1]

        if roi_large.size == 0:
            # print("Debug: get_rotated_roi: roi_large extraite est vide.")
            return None, None

        # Ajuster le centre de rotation pour être relatif à roi_large
        new_center_x_for_warp = center_x - clip_x0
        new_center_y_for_warp = center_y - clip_y0

        # Créer la matrice de rotation pour la ROI large
        roi_rotation_matrix = cv2.getRotationMatrix2D(
            (new_center_x_for_warp, new_center_y_for_warp), angle, 1.0
        )

        # Appliquer la rotation à la ROI large
        rotated_roi_large = cv2.warpAffine(roi_large, roi_rotation_matrix,
                                           (roi_large.shape[1], roi_large.shape[0]),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Extraire la partie centrale de taille (w, h) du rotated_roi_large
        final_roi_x = int(new_center_x_for_warp - w // 2)
        final_roi_y = int(new_center_y_for_warp - h // 2)

        final_roi_x_end = final_roi_x + w
        final_roi_y_end = final_roi_y + h

        # Limiter aux bords de rotated_roi_large
        final_roi_x = max(0, final_roi_x)
        final_roi_y = max(0, final_roi_y)
        final_roi_x_end = min(rotated_roi_large.shape[1], final_roi_x_end)
        final_roi_y_end = min(rotated_roi_large.shape[0], final_roi_y_end)

        final_roi = rotated_roi_large[final_roi_y:final_roi_y_end, final_roi_x:final_roi_x_end]

        # Redimensionner si la taille finale n'est pas correcte (peut arriver si le recadrage a coupé)
        if final_roi.shape[0] != h or final_roi.shape[1] != w:
            if w > 0 and h > 0:  # S'assurer que les dimensions de redimensionnement sont valides
                final_roi = cv2.resize(final_roi, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # print(f"Debug: get_rotated_roi: Impossible de redimensionner à w={w}, h={h}")
                return None, None

        # Retourne la ROI finale et ses coordonnées de coin supérieur gauche dans l'image originale
        return final_roi, (clip_x0 + final_roi_x, clip_y0 + final_roi_y)

    def search_region_rbb(self, gray, obj, scale, search_area):
        """Search with RBB support and drift protection"""
        if scale not in obj["templates"]:
            return -1, None, 0

        template = obj["templates"][scale]
        T6, T7 = template['T6'], template['T7']
        w, h = template['size']

        x0, x1, y0, y1 = search_area
        best_score = -1
        best_coords = None
        best_angle = obj["angle"]  # Commence la recherche autour de l'angle actuel de l'objet

        # Step size basé sur la confiance
        base_step_x = max(1, w // 10)  # Pas plus grand pour la position
        base_step_y = max(1, h // 10)

        # Réduire le pas si la confiance est faible pour une recherche plus fine
        pos_step_x = base_step_x if obj["confidence"] > 0.5 else max(1, base_step_x // 2)
        pos_step_y = base_step_y if obj["confidence"] > 0.5 else max(1, base_step_y // 2)

        # Angles à tester (centré sur l'angle actuel)
        current_angle = obj["angle"]
        angle_range = 15 if obj["confidence"] > 0.7 else 30  # Plage de recherche d'angle
        angle_step = 5  # Pas de recherche d'angle

        angles_to_test = [self._normalize_angle(current_angle + da) for da in
                          range(-angle_range, angle_range + angle_step, angle_step)]

        # S'assurer que l'angle actuel est toujours testé
        if current_angle not in angles_to_test:
            angles_to_test.append(current_angle)
            angles_to_test = sorted(list(set(angles_to_test)))  # Supprimer les doublons et trier

        for angle in angles_to_test:
            for y in range(y0, min(y1, gray.shape[0] - h), pos_step_y):
                for x in range(x0, min(x1, gray.shape[1] - w), pos_step_x):
                    center_x_candidate = x + w // 2
                    center_y_candidate = y + h // 2

                    # Extraire ROI avec rotation
                    roi, _ = self.get_rotated_roi(gray, center_x_candidate, center_y_candidate, w, h, angle)
                    if roi is None:
                        continue

                    I6 = extract_bit_plane_numba(roi, 6)
                    I7 = extract_bit_plane_numba(roi, 7)

                    # Score booléen
                    zeros = compute_psi_numba(T6, T7, I6, I7)

                    # Optimisation: si le score booléen est trop faible, passer
                    if zeros < best_score * 0.7 and best_score != -1:  # Un peu moins strict pour ne pas rater
                        continue

                    # 🛡️ PROTECTION CONTRE LA DÉRIVE
                    roi_hist = compute_weighted_histogram_optimized(roi)

                    current_hist_sim = cv2.compareHist(obj["current_hist"], roi_hist, cv2.HISTCMP_CORREL)
                    original_hist_sim = cv2.compareHist(obj["original_hist"], roi_hist, cv2.HISTCMP_CORREL)

                    hist_sim = current_hist_sim  # Par défaut, utilise l'historique courant

                    # Si le template original donne un bien meilleur score, on a probablement dérivé
                    if (obj["drift_protection"] and
                            original_hist_sim > current_hist_sim + obj["original_score_threshold"]):
                        hist_sim = original_hist_sim
                        # print(f"Retour au template original pour {obj['name']} (dérive détectée)")
                        obj["current_hist"] = obj["original_hist"].copy()  # Réinitialiser le template courant

                    total_score = 0.7 * zeros + 0.3 * max(0, hist_sim) * 1000

                    if total_score > best_score:
                        best_score = total_score
                        best_coords = (center_x_candidate, center_y_candidate, w, h)
                        best_angle = angle

        return best_score, best_coords, best_angle

    def update_template_safe(self, gray, obj, center_x, center_y, w, h, angle):
        """Mise à jour sécurisée du template avec vérification de dérive"""
        obj["update_count"] += 1

        # Mise à jour moins fréquente et plus conservative
        if obj["update_count"] % 50 == 0 and obj["confidence"] > 0.8:  # Augmenté la confiance requise
            roi, _ = self.get_rotated_roi(gray, center_x, center_y, w, h, angle)
            if roi is not None and roi.shape[0] == h and roi.shape[1] == w:  # Vérifier la taille exacte
                new_hist = compute_weighted_histogram_optimized(roi)

                # Vérifier la similarité avec le template original avant mise à jour
                original_sim = cv2.compareHist(obj["original_hist"], new_hist, cv2.HISTCMP_CORREL)

                # Ne mettre à jour que si on reste proche du template original
                if original_sim > 0.6:  # Seuil de sécurité augmenté
                    alpha = 0.05  # Mise à jour très douce
                    obj["current_hist"] = (1 - alpha) * obj["current_hist"] + alpha * new_hist
                    # print(f"Template mis à jour prudemment pour {obj['name']}")
                else:
                    # print(f"Mise à jour refusée pour {obj['name']} (trop différent de l'original)")
                    pass  # Ne pas imprimer constamment
            # else:
            # print(f"Debug: update_template_safe: ROI invalide ou taille incorrecte pour {obj['name']}")

    def track_object(self, gray, obj):
        """Track with RBB support and drift protection"""
        best_score = -1
        best_coords = (obj["center_x"], obj["center_y"], obj["w0"], obj["h0"])
        best_angle = obj["angle"]  # Initialiser avec l'angle actuel de l'objet
        found = False

        # Rayon de recherche adaptatif
        base_radius = max(30, max(obj["w0"], obj["h0"]) // 2)
        search_radius = min(150, int(base_radius * (2.5 - obj["confidence"])))  # Rayon max augmenté

        local_area = (
            max(0, obj["center_x"] - search_radius),
            min(gray.shape[1], obj["center_x"] + search_radius),
            max(0, obj["center_y"] - search_radius),
            min(gray.shape[0], obj["center_y"] + search_radius)
        )

        # Recherche locale avec RBB
        for scale in self.scales:
            score, coords, angle = self.search_region_rbb(gray, obj, scale, local_area)
            if score > best_score:
                best_score = score
                if coords:
                    best_coords = coords
                    best_angle = angle

        if best_score > self.score_threshold:
            found = True
        else:
            # Recherche globale limitée
            obj["lost_frames"] += 1
            if obj["lost_frames"] < 10 and obj["confidence"] > 0.1:  # Confiance minimale pour la recherche globale
                global_area = (0, gray.shape[1], 0, gray.shape[0])
                score, coords, angle = self.search_region_rbb(gray, obj, 1.0, global_area)
                if score > best_score:
                    best_score = score
                    if coords:
                        best_coords = coords
                        best_angle = angle
                        found = True

        # Mise à jour de l'état
        if found:
            center_x, center_y, w, h = best_coords
            obj["center_x"], obj["center_y"] = center_x, center_y

            # Lissage de l'angle
            obj["angle_history"].append(best_angle)
            if len(obj["angle_history"]) > self.angle_history_len:
                obj["angle_history"].pop(0)

            # Calculer la moyenne des angles, en tenant compte du wrap-around si nécessaire
            # Pour des angles dans [0, 360), une moyenne directe est souvent suffisante pour de petits mouvements
            # Pour des mouvements plus grands, il faudrait une moyenne circulaire.
            # Pour l'instant, une moyenne simple est appliquée, car search_region_rbb cherche déjà localement.
            obj["angle"] = np.mean(obj["angle_history"])
            obj["angle"] = self._normalize_angle(obj["angle"])  # Normaliser l'angle lissé

            obj["confidence"] = min(1.0, obj["confidence"] + 0.15)
            obj["lost_frames"] = 0

            self.update_template_safe(gray, obj, center_x, center_y, w, h, obj["angle"])
        else:
            obj["confidence"] = max(0.05, obj["confidence"] - 0.1)

        return best_coords, best_score, found, obj["angle"]

    def get_rbb_corners(self, center_x, center_y, w, h, angle):
        """Calcule les coins du rectangle rotaté pour l'affichage"""
        # Coins du rectangle centré sur l'origine
        corners = np.array([
            [-w // 2, -h // 2],
            [w // 2, -h // 2],
            [w // 2, h // 2],
            [-w // 2, h // 2]
        ], dtype=np.float32)

        # Matrice de rotation
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        # Appliquer rotation et translation
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += center_x
        rotated_corners[:, 1] += center_y

        return rotated_corners.astype(np.int32)


class OptimizedObjectTracker:
    def __init__(self, history=300, var_threshold=50):
        self.cap = None
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True
        )
        self.tracker = OptimizedTracker()
        self.frame_skip = 0
        self.frame_count = 0

        self.detection_enabled = True
        self.show_detections = True
        self.detection_frame_interval = 1

    def setup_camera(self, source=0):
        """Setup camera with optimized settings"""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def point_in_rotated_rect(self, point, box_points):
        """Vérifie si un point est dans un rectangle rotaté"""
        # box_points doit être un tableau numpy de forme (4, 2)
        return cv2.pointPolygonTest(box_points, (float(point[0]), float(point[1])), False) >= 0

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback avec extraction du centre et de l'angle"""
        if event == cv2.EVENT_LBUTTONDOWN:
            rotated_boxes = param.get("rotated_boxes", [])
            rotated_rects = param.get("rotated_rects", [])
            frame = param.get("frame")

            if frame is None:
                print("Debug: Frame is None in on_mouse.")
                return

            for i, box_points in enumerate(rotated_boxes):
                # Assurez-vous que box_points est valide avant de l'utiliser
                if box_points is None or box_points.shape != (4, 2):
                    continue

                if self.point_in_rotated_rect((x, y), box_points):
                    if i < len(rotated_rects):
                        rect_info = rotated_rects[i]
                        center = rect_info[0]
                        size = rect_info[1]
                        angle_raw = rect_info[2]  # Angle brut de minAreaRect

                        center_x, center_y = int(center[0]), int(center[1])
                        w, h = int(size[0]), int(size[1])

                        # Normaliser l'angle de minAreaRect pour le suivi
                        # minAreaRect angle convention: [-90, 0) for width as longer side, or height as longer side
                        # Let's normalize to [0, 360) for consistency
                        normalized_angle = angle_raw
                        if w < h:  # If height is the longer side, add 90 degrees to align with width
                            normalized_angle += 90
                        normalized_angle = self.tracker._normalize_angle(normalized_angle)

                        roi, _ = self.tracker.get_rotated_roi(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                            center_x, center_y, w, h, normalized_angle
                        )

                        if roi is None:
                            print("Impossible d'extraire la ROI rotatée pour le tracker (clic).")
                            continue

                        name = f"Object_{len(self.tracker.trackers) + 1}"
                        success = self.tracker.add_tracker(name, roi, center_x, center_y, w, h, normalized_angle)

                        if success:
                            print(
                                f"{name} sélectionné (centre: {center_x},{center_y}, angle: {normalized_angle:.1f}°)")
                            if len(self.tracker.trackers) >= 1:
                                print("💡 Appuyez sur 'd' pour désactiver la détection")
                        else:
                            print(f"Impossible de tracker {name}")
                        break

    def detect_objects(self, frame):
        """Detect moving objects with RBB"""
        if not self.detection_enabled:
            return [], []

        mask = self.object_detector.apply(frame)

        if mask is None or mask.size == 0:
            return [], []

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rotated_boxes = []  # Pour l'affichage (les 4 points)
        rotated_rects = []  # Pour les infos complètes (center, size, angle)

        for cnt in contours:
            if len(cnt) < 5:  # Un contour doit avoir au moins 5 points pour fitEllipse, 3 pour minAreaRect
                continue

            area = cv2.contourArea(cnt)
            if 100 < area < 50000:
                rect = cv2.minAreaRect(cnt)

                center_pt, size, angle_raw = rect
                width, height = size

                if width == 0 or height == 0:
                    continue

                # Normaliser l'angle ici aussi pour le stocker dans rotated_rects
                normalized_angle_for_storage = angle_raw
                if width < height:
                    normalized_angle_for_storage += 90
                normalized_angle_for_storage = self.tracker._normalize_angle(normalized_angle_for_storage)

                aspect_ratio = max(width, height) / min(width, height)
                x_aabb, y_aabb, w_aabb, h_aabb = cv2.boundingRect(cnt)
                extent = area / (w_aabb * h_aabb) if w_aabb * h_aabb > 0 else 0

                if (aspect_ratio < 5.0 and extent > 0.3 and min(width, height) > 20):
                    box_points = cv2.boxPoints(rect)
                    box_points = np.int32(box_points)
                    rotated_boxes.append(box_points)
                    # Stocker le rect original MAIS avec l'angle normalisé
                    rotated_rects.append((center_pt, size, normalized_angle_for_storage))

        return rotated_boxes, rotated_rects

    def toggle_detection(self):
        """Active/désactive la détection d'objets"""
        self.detection_enabled = not self.detection_enabled
        status = "ACTIVÉE" if self.detection_enabled else "DÉSACTIVÉE"
        cpu_info = "consommation CPU normale" if self.detection_enabled else "économie CPU"
        print(f"Détection {status} ({cpu_info})")

    def run(self):
        """Main loop avec RBB tracking complet"""
        if not self.cap:
            # self.cap = cv2.VideoCapture("images/templates/childs.mp4") # Pour tester avec une vidéo
            self.setup_camera()

        cv2.namedWindow("RBB Object Tracking", cv2.WINDOW_AUTOSIZE)

        print("Instructions:")
        print("- Clic sur objet détecté (contour vert) pour tracker avec RBB")
        print("- 'd' : ACTIVER/DÉSACTIVER détection")
        print("- 'v' : basculer affichage détections")
        print("- 's' : ajuster frame skip")
        print("- 'p' : activer/désactiver protection dérive")
        print("- 'r' : reset trackers")
        print("- ESC : quitter")

        last_rotated_boxes = []
        last_rotated_rects = []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Fin de la vidéo ou erreur de lecture de la caméra.")
                break

            self.frame_count += 1

            if self.frame_count % (self.frame_skip + 1) != 0:
                continue

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Détection conditionnelle
            if self.detection_enabled and self.frame_count % self.detection_frame_interval == 0:
                rotated_boxes, rotated_rects = self.detect_objects(frame)
                last_rotated_boxes = rotated_boxes
                last_rotated_rects = rotated_rects
            else:
                # Si détection désactivée ou frame_skip, utiliser les dernières détections pour l'affichage
                rotated_boxes = last_rotated_boxes
                rotated_rects = last_rotated_rects

            # Affichage des détections RBB
            if self.show_detections:
                for box_points in rotated_boxes:
                    color = (0, 255, 0) if self.detection_enabled else (128, 128, 128)
                    cv2.drawContours(display_frame, [box_points], 0, color, 2)

                    if self.detection_enabled:
                        # Calculer le centre pour le texte "Click"
                        if len(box_points) > 0:
                            center = np.mean(box_points, axis=0).astype(int)
                            cv2.putText(display_frame, "Click", tuple(center - [20, -5]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Callback souris avec infos RBB complètes
            cv2.setMouseCallback("RBB Object Tracking", self.on_mouse,
                                 param={
                                     "frame": frame.copy(),
                                     "rotated_boxes": rotated_boxes,
                                     "rotated_rects": rotated_rects
                                 })

            # TRACKING AVEC RBB
            active_trackers = []
            for obj in self.tracker.trackers:
                coords, score, found, angle = self.tracker.track_object(gray, obj)

                if found or obj["confidence"] > 0.05:  # Afficher même avec très faible confiance pour voir la dérive
                    center_x, center_y, w, h = coords

                    # Couleur selon l'état
                    if found and obj["confidence"] > 0.7:
                        color = (0, 255, 0)  # Vert (tracké et confiant)
                    elif found:
                        color = (255, 165, 0)  # Orange (tracké mais moins confiant)
                    else:
                        color = (0, 0, 255)  # Rouge (perdu ou très faible confiance)

                    # AFFICHAGE RBB POUR LE TRACKING
                    rbb_corners = self.tracker.get_rbb_corners(center_x, center_y, w, h, angle)
                    cv2.drawContours(display_frame, [rbb_corners], 0, color, 3)

                    # Croix au centre
                    cv2.drawMarker(display_frame, (center_x, center_y), color,
                                   cv2.MARKER_CROSS, 10, 2)

                    # Informations détaillées
                    info = f"{obj['name']}"
                    confidence_info = f"Conf: {obj['confidence']:.2f}"
                    angle_info = f"Angle: {angle:.1f}°"
                    score_info = f"Score: {score:.0f}"

                    # Positionnement du texte
                    text_offset_y = h // 2 + 15
                    cv2.putText(display_frame, info, (center_x - w // 2, center_y - h // 2 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(display_frame, confidence_info, (center_x - w // 2, center_y - h // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(display_frame, angle_info, (center_x - w // 2, center_y - h // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(display_frame, score_info, (center_x - w // 2, center_y + text_offset_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Indicateur de protection anti-dérive
                    if obj["drift_protection"]:
                        cv2.putText(display_frame, "🛡️", (center_x + w // 2 - 20, center_y - h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    active_trackers.append(obj)
                else:
                    if obj["lost_frames"] > 60:
                        print(f"🗑️ Suppression du tracker: {obj['name']}")
                    # else:
                    # active_trackers.append(obj) # Ne pas ajouter si perdu et pas encore supprimé

            self.tracker.trackers = active_trackers

            # Informations de statut
            info_y = 30
            status_color = (0, 255, 0) if self.detection_enabled else (0, 0, 255)
            detection_status = "ON" if self.detection_enabled else "OFF"

            cv2.putText(display_frame, f"RBB Trackers: {len(self.tracker.trackers)}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Détection: {detection_status}",
                        (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display_frame, f"Objets détectés: {len(rotated_boxes)}",
                        (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not self.detection_enabled and len(self.tracker.trackers) > 0:
                cv2.putText(display_frame, "RBB CPU SAVED MODE",
                            (display_frame.shape[1] - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RBB Object Tracking", display_frame)

            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('d'):
                self.toggle_detection()
            elif key == ord('v'):
                self.show_detections = not self.show_detections
                status = "visible" if self.show_detections else "masqué"
                print(f"Affichage détections: {status}")
            elif key == ord('s'):
                self.frame_skip = (self.frame_skip + 1) % 4
                print(f"Frame skip: {self.frame_skip}")
            elif key == ord('p'):  # Toggle drift protection
                for obj in self.tracker.trackers:
                    obj["drift_protection"] = not obj["drift_protection"]
                status = "ACTIVATE" if (self.tracker.trackers and
                                       self.tracker.trackers[0]["drift_protection"]) else "OFF"
                print(f"Protection: {status}")
            elif key == ord('r'):
                self.tracker.trackers.clear()
                if not self.detection_enabled:
                    self.detection_enabled = True
                    print("detection on")
                print("disable all trackers")

        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Clean done")


if __name__ == "__main__":
    try:
        tracker = OptimizedObjectTracker()
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
