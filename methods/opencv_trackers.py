import cv2
from methods.base import TrackerMethod

class OpenCVTracker(TrackerMethod):
    def __init__(self, init_frame, roi, tracker_type="MOSSE"):
        self.tracker_type = tracker_type.upper()

        tracker_creators = {
            "MOSSE": ["TrackerMOSSE_create", "legacy.TrackerMOSSE_create"],
            "KCF": ["TrackerKCF_create", "legacy.TrackerKCF_create"],
            "CSRT": ["TrackerCSRT_create", "legacy.TrackerCSRT_create"],
        }

        if self.tracker_type not in tracker_creators:
            raise ValueError("Unsupported tracker type")

        tracker_obj = None
        for creator_name in tracker_creators[self.tracker_type]:
            try:
                module = cv2
                for part in creator_name.split("."):
                    module = getattr(module, part)
                tracker_obj = module()
                break
            except AttributeError:
                continue

        if tracker_obj is None:
            raise RuntimeError(f"Tracker {self.tracker_type} not available in this OpenCV build")

        self.tracker = tracker_obj
        try:
            x, y, w, h = map(int, roi)
        except Exception as e:
            raise ValueError(f"Erreur lors de la conversion du ROI : {roi} ({e})")

        # Optionnel : affiche pour debug
        print(f"[DEBUG] ROI utilisé pour init: {(x, y, w, h)} -- types: {[type(x), type(y), type(w), type(h)]}")

        # Init le tracker
        self.tracker.init(init_frame, (x, y, w, h))

    def update(self, frame):
        """Retourne (bbox, score) comme les autres trackers."""
        ok, bbox = self.tracker.update(frame)
        if ok:
            x, y, w, h = bbox
            return (int(x), int(y), int(w), int(h)), 1.0  # score=1.0 si trouvé
        else:
            return None, 0.0  # perdu
