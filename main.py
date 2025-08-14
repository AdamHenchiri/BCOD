import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from methods.bitplane_hist import BitplaneHistTracker


def get_rotated_rect(cx, cy, w, h, angle_deg):
    rect = ((cx, cy), (w, h), angle_deg)
    box = cv2.boxPoints(rect)
    return np.int32(box)


class MultiThreadedBitplaneManager:

    def __init__(self, source=0, max_workers=4):
        self.source = source
        self.cap = None
        self.max_workers = max_workers
        self.trackers = []
        self.paused = False
        self.current_points = []  # list of (x,y)
        self.last_frame = None
        self.frame_skip = 0
        self.frame_count = 0
        self.main_window = "Bitplane Multi Tracker"
        self.selection_window = "Object Selection"
        self.selection_window_open = False
        self.selection_frame = None
        cv2.namedWindow(self.main_window)
        cv2.setMouseCallback(self.main_window, self.on_mouse_main)

        self.CONFIDENCE_THRESHOLD_GOOD = 10000
        self.CONFIDENCE_THRESHOLD_WARNING = 5000
        self.CONFIDENCE_THRESHOLD_BAD = 260

    def on_mouse_main(self, event, x, y, flags, param):
        pass
    def setup_camera(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source {self.source}")

    def on_mouse_selection(self, event, x, y, flags, param):
        """Mouse callback pour la fenêtre de sélection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            print(f"[coord] : {x},{y} ({len(self.current_points)} out of 4)")

            if len(self.current_points) == 4:
                pts = np.array(self.current_points, dtype=np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                if self.selection_frame is None:  # MODIFIÉ: utilise selection_frame
                    print("!!! frame not available !!!")
                    self.current_points.clear()
                    return

                H, W = self.selection_frame.shape[:2]  # MODIFIÉ: utilise selection_frame
                x0 = max(0, bx)
                y0 = max(0, by)
                x1 = min(W, bx + bw)
                y1 = min(H, by + bh)
                if (x1 - x0) < 8 or (y1 - y0) < 8:
                    print("!!! ROI is too small !!!")
                    self.current_points.clear()
                    return

                roi = self.selection_frame[y0:y1, x0:x1].copy()  # MODIFIÉ: utilise selection_frame
                roi_tuple = (x0, y0, x1 - x0, y1 - y0)

                try:
                    t = BitplaneHistTracker(self.selection_frame, roi_tuple)  # MODIFIÉ: utilise selection_frame
                    # ... reste du code de création du tracker identique ...
                    self.trackers.append(t)
                    print(f"[ADD] Tracker added: {t.name} bbox={roi_tuple}")
                except Exception as e:
                    print(f"!!! ERROR : {e} !!!")

                # NOUVEAU: Fermer la fenêtre et reprendre
                self.close_selection_window()

    def open_selection_window(self):
        """Ouvre la fenêtre de sélection et met en pause"""
        if self.last_frame is None:
            print("!!! No frame available for selection !!!")
            return

        self.selection_frame = self.last_frame.copy()
        self.current_points.clear()
        self.selection_window_open = True
        self.paused = True  # PAUSE la vidéo

        cv2.namedWindow(self.selection_window)
        cv2.setMouseCallback(self.selection_window, self.on_mouse_selection)

    def close_selection_window(self):
        """Ferme la fenêtre de sélection et reprend la vidéo"""
        if self.selection_window_open:
            cv2.destroyWindow(self.selection_window)
            self.selection_window_open = False
            self.current_points.clear()
            self.selection_frame = None
            self.paused = False

    def update_selection_window(self):
        """Met à jour l'affichage de la fenêtre de sélection"""
        if not self.selection_window_open or self.selection_frame is None:
            return

        display = self.selection_frame.copy()

        # Dessiner les points de sélection
        if self.current_points:
            for i, (px, py) in enumerate(self.current_points):
                cv2.circle(display, (px, py), 4, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)

            if len(self.current_points) >= 2:
                pts = np.array(self.current_points, dtype=np.int32)
                if len(self.current_points) >= 3:
                    cv2.polylines(display, [pts], True, (0, 255, 255), 2)
                else:
                    cv2.polylines(display, [pts], False, (0, 255, 255), 2)

        cv2.putText(display, f"Select 4 points: {len(self.current_points)}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.selection_window, display)

    def run(self):
        if not self.cap:
            try:
                self.setup_camera()
            except RuntimeError as e:
                print(f"!!! ERROR : {e} !!!")
                self.cleanup()
                return
        print("CONTRÔLES:")
        print("  SPACE = pause/resume (you can select 4 points in pause mode)")
        print("  C = delete current selection points")
        print("  S = changer frame skip (0,1,2,3)")
        print("  R = deleter all trackers (reset selection)")
        print("  ESC = exit")

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("!!! ERROR : Cannot read frame from source !!!")
                    break

                self.frame_count += 1
                self.last_frame = frame.copy()
            else:
                frame = self.last_frame.copy() if self.last_frame is not None else None
                if frame is None:
                    continue

            # this condition skips frames based on frame_skip value
            if self.frame_count % (self.frame_skip + 1) != 0:
                display = frame.copy()
                self._draw_overlay(display)
                cv2.imshow("Bitplane Multi Tracker", display)
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)
                continue

            display = frame.copy()

            # UPDATE trackers in parallel (BitplaneHistTracker.update(frame) -> (bbox, score))
            active = []
            if self.trackers:
                workers = min(self.max_workers, max(1, len(self.trackers)))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(t.update, frame): t for t in self.trackers}
                    for fut in as_completed(futures):
                        t = futures[fut]
                        try:
                            res = fut.result(timeout=2.0)
                        except Exception as e:
                            print(f"[track] exception from tracker {getattr(t, 'name', '?')}: {e}")
                            continue
                        if not res:
                            continue
                        bbox, score = res  # bbox expected as (x,y,w,h)
                        if bbox is None:
                            t.lost_frames = getattr(t, 'lost_frames', 0) + 1
                        else:
                            t.lost_frames = 0
                            t.last_box = tuple(map(int, bbox))
                            t.last_score = score

                        if hasattr(t, 'last_box') and t.last_box:
                            x, y, w, h = t.last_box
                            lost_frames = getattr(t, 'lost_frames', 0)
                            if lost_frames > 5:
                                color = (0, 0, 255)  # Red = lost
                            elif score is None:
                                color = (0, 100, 255)  # light Orange = score not available
                            elif score > self.CONFIDENCE_THRESHOLD_GOOD:
                                color = (0, 255, 0)  # Green = good score
                            elif score > self.CONFIDENCE_THRESHOLD_WARNING:
                                color = (0, 255, 255)  # Yellow = medium scor
                            else:
                                color = (0, 165, 255)  # Orange = low score

                            cx, cy = x + w // 2, y + h // 2

                            # Dessin BB avec rotation si dispo
                            if hasattr(t, "last_angle") and t.last_angle is not None:
                                box = get_rotated_rect(cx, cy, w, h, t.last_angle)
                                cv2.polylines(display, [box], True, color, 2)
                            else:
                                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

                            # Marque le centre
                            cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 10, 2)

                            cv2.putText(display, getattr(t, 'name', 'Tracker'), (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if score is not None:
                                info_text = f"Score:{int(score)}"
                            else:
                                info_text = "Score:NA"
                            if lost_frames > 0:
                                info_text += f" Lost:{lost_frames}"
                            cv2.putText(display, info_text,
                                        (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                        keep_tracker = True
                        lost_frames = getattr(t, 'lost_frames', 0)

                        if lost_frames > 30:  # lost for too long
                            keep_tracker = False
                            reason = f"lost {lost_frames} frames"
                        elif score is not None and score < self.CONFIDENCE_THRESHOLD_BAD:
                            # store in history
                            low_score_count = getattr(t, 'low_score_count', 0) + 1
                            t.low_score_count = low_score_count
                            if low_score_count > 10:  # 10 frames consécutives avec un mauvais score
                                keep_tracker = False
                                reason = f"low score {int(score)} for {low_score_count} frames"
                        else:
                            t.low_score_count = 0

                        if keep_tracker:
                            active.append(t)
                        else:
                            print(f"[remove] {getattr(t, 'name', '?')} removed ({reason})")

                        # commit trackers list
                        self.trackers = active

            # draw selection points if paused
            if self.paused and self.current_points:
                for i, (px, py) in enumerate(self.current_points):
                    cv2.circle(display, (px, py), 4, (0, 255, 255), -1)
                    cv2.putText(display, str(i + 1), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)

                if len(self.current_points) >= 2:
                    pts = np.array(self.current_points, dtype=np.int32)
                    if len(self.current_points) >= 3:
                        cv2.polylines(display, [pts], True, (0, 255, 255), 2)
                    else:
                        cv2.polylines(display, [pts], False, (0, 255, 255), 2)

            # final overlay and show
            self._draw_overlay(display)
            cv2.imshow("Bitplane Multi Tracker", display)
            if self.selection_window_open:
                self.update_selection_window()
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)

        self.cleanup()

    def _draw_overlay(self, display):
        # Info en haut à gauche
        cv2.putText(display, f"Actif trackers : {len(self.trackers)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        if self.selection_window_open:
            status_text = "PAUSED - Selection mode active"
            status_color = (0, 255, 255)
        elif self.paused:
            status_text = "PAUSED"
            status_color = (0, 255, 255)
        else:
            status_text = "Running : tracking active"
            status_color = (0, 255, 0)
        cv2.putText(display, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)


        status_text = "Break : you can select 4 points" if self.paused else "Running : tracking active"
        status_color = (0, 255, 255) if self.paused else (0, 255, 0)
        cv2.putText(display, status_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        if self.paused and self.current_points:
            cv2.putText(display, f"Points: {len(self.current_points)} out of 4", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

        if self.frame_skip > 0:
            cv2.putText(display, f"Skip: {self.frame_skip}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _handle_key(self, key):
        if key == 27:  # ESC
            if self.selection_window_open:
                self.close_selection_window()
                print("Selection cancelled")
            else:
                self.cleanup()
                exit(0)
        elif key == ord(' '):
            if not self.selection_window_open:
                self.open_selection_window()
        elif key == ord('c'):
            self.current_points.clear()
            print("clear current selection points")
        elif key == ord('s'):
            self.frame_skip = (self.frame_skip + 1) % 4
            print(f"frame_skip = {self.frame_skip}")
        elif key == ord('r'):
            self.trackers.clear()
            print("clear all trackers and reset selection points")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("cleanup done")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0 for webcam, or path to video file or camera stream URL")
    p.add_argument("--workers", type=int, default=4, help="Number of worker threads for tracking (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src = 0 if args.source == "0" else args.source
    mgr = MultiThreadedBitplaneManager(source=src, max_workers=args.workers)
    try:
        mgr.run()
    except KeyboardInterrupt:
        print("Exiting by user request...")
    except Exception as e:
        print("ERROR :", e)
        import traceback;

        traceback.print_exc()
    finally:
        mgr.cleanup()