import cv2
import numpy as np
from utils import extract_bit_plane_numba, compute_psi_numba


class OptimizedTracker:
    """Tracker basé sur bitplanes et corrélation logique."""
    def __init__(self):
        self.trackers = []
        self.scales = [0.8, 1.0, 1.2]
        self.score_threshold = 2750  # seuil de détection

    def add_tracker(self, name, roi, x, y, w, h):
        """Ajoute un nouveau tracker basé sur la ROI fournie (grayscale)."""
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

        if gray_roi.size == 0 or w < 10 or h < 10:
            print(f"⚠️ ROI trop petite pour {name}, ignorée")
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
            print(f"⚠️ Impossible de créer les templates pour {name}")
            return False

        tracker = {
            "name": name,
            "templates": templates,
            "w0": w,
            "h0": h,
            "last_x": x,
            "last_y": y,
            "confidence": 1.0,
            "lost_frames": 0
        }
        self.trackers.append(tracker)
        print(f"✅ {name} ajouté au tracking!")
        return True

    def search_region(self, gray, obj, scale, search_area):
        """Recherche du meilleur match dans une zone."""
        if scale not in obj["templates"]:
            return -1, None

        template = obj["templates"][scale]
        T6, T7 = template['T6'], template['T7']
        w, h = template['size']

        x0, x1, y0, y1 = search_area
        best_score = -1
        best_coords = None
        step = max(1, min(4, int(4 * (1.5 - obj["confidence"]))))

        for y in range(y0, min(y1, gray.shape[0] - h), step):
            for x in range(x0, min(x1, gray.shape[1] - w), step):
                roi = gray[y:y + h, x:x + w]
                I6 = extract_bit_plane_numba(roi, 6)
                I7 = extract_bit_plane_numba(roi, 7)

                score = compute_psi_numba(T6, T7, I6, I7)
                if score > best_score:
                    best_score = score
                    best_coords = (x, y, w, h)

        return best_score, best_coords

    def track_object(self, gray, obj):
        """Met à jour la position d'un objet suivi."""
        best_score = -1
        best_coords = (obj["last_x"], obj["last_y"], obj["w0"], obj["h0"])
        found = False

        # Recherche locale
        search_radius = max(30, int(60 * (2.0 - obj["confidence"])))
        local_area = (
            max(0, obj["last_x"] - search_radius),
            min(gray.shape[1], obj["last_x"] + search_radius),
            max(0, obj["last_y"] - search_radius),
            min(gray.shape[0], obj["last_y"] + search_radius)
        )

        for scale in self.scales:
            score, coords = self.search_region(gray, obj, scale, local_area)
            if score > best_score:
                best_score = score
                if coords:
                    best_coords = coords

        if best_score > self.score_threshold:
            found = True
        else:
            obj["lost_frames"] += 1
            obj["confidence"] = max(0.1, obj["confidence"] - 0.15)

        if found:
            x, y, w, h = best_coords
            obj["last_x"], obj["last_y"] = x, y
            obj["confidence"] = min(1.0, obj["confidence"] + 0.1)
            obj["lost_frames"] = 0

        return best_coords, best_score, found


class ObjectTrackerOBB:
    """Gestion de la capture vidéo et du suivi d'objets via OBB manuelles."""
    def __init__(self):
        self.cap = None
        self.tracker = OptimizedTracker()
        self.frame_count = 0
        self.paused = False
        self.current_points = []
        self.last_frame = None

    def setup_camera(self, source=0):
        """Initialise la caméra."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def on_mouse(self, event, x, y, flags, param):
        """Capture des 4 points pour OBB."""
        if self.paused and event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            print(f"Point ajouté: {x},{y}")

            if len(self.current_points) == 4:
                pts = np.array(self.current_points, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                roi = self.last_frame[y:y + h, x:x + w]
                name = f"Object_{len(self.tracker.trackers)+1}"
                self.tracker.add_tracker(name, roi, x, y, w, h)
                self.current_points = []

    def run(self):
        """Boucle principale."""
        self.setup_camera(0)
        cv2.namedWindow("OBB Tracker")
        cv2.setMouseCallback("OBB Tracker", self.on_mouse)

        print("🎯 Mode OBB activé :")
        print("- Espace : pause/reprise pour sélectionner un objet")
        print("- Cliquez 4 points pour définir une OBB")
        print("- R : reset trackers")
        print("- ESC : quitter")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_count += 1
            else:
                frame = frame  # garde l'image de pause

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Affichage des OBB en cours
            if len(self.current_points) > 0:
                for i in range(len(self.current_points)):
                    cv2.circle(display_frame, self.current_points[i], 3, (0, 255, 255), -1)
                    if i > 0:
                        cv2.line(display_frame, self.current_points[i-1], self.current_points[i], (0, 255, 255), 2)

            # Tracking
            active_trackers = []
            for obj in self.tracker.trackers:
                coords, score, found = self.tracker.track_object(gray, obj)
                x, y, w, h = coords
                color = (0, 255, 0) if found else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"{obj['name']} S:{score:.0f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if found or obj["confidence"] > 0.1:
                    active_trackers.append(obj)

            self.tracker.trackers = active_trackers

            self.last_frame = frame.copy()
            cv2.imshow("OBB Tracker", display_frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print("break" if self.paused else "continue")
            elif key == ord('r'):
                self.tracker.trackers.clear()
                print("clear trackers")

        self.cleanup()

    def cleanup(self):
        """Libère les ressources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("clean done")


if __name__ == "__main__":
    tracker = ObjectTrackerOBB()
    tracker.run()
