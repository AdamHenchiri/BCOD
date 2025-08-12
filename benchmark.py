import cv2
import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool, Lock, Manager
from YTBBLoader import YTBBLoader
from evaluate import compute_iou, compute_accuracy
from methods.opencv_trackers import OpenCVTracker
from methods.orb_match import ORBTracker
from methods.template_match import TemplateMatchTracker
from methods.bitplane import BitplaneTracker
from methods.bitplane_hist import BitplaneHistTracker
from methods.homography import HomographyTracker
from methods.bitplane_ens import BitplaneEnsembleTracker

# -------------------------------Paramètres globaux
csv_path = "ytbb_subset.csv"
videos_folder = "yt_videos"
MAX_FRAMES = 300
N_PROCESSES = 4
SHOW_PREVIEW = False
OUTPUT_CSV = "benchmark_results.csv"
BATCH_SIZE = 2

def create_template_tracker(frame, roi):
    return TemplateMatchTracker(frame, roi)

def create_bitplane_tracker(frame, roi):
    return BitplaneTracker(frame, roi)

def create_bitplane_hist_tracker(frame, roi):
    return BitplaneHistTracker(frame, roi)

def create_homography_tracker(frame, roi):
    return HomographyTracker(frame, roi)

def create_mosse_tracker(init_frame, roi):
    return OpenCVTracker(init_frame, roi, tracker_type="MOSSE")

def create_kcf_tracker(init_frame, roi):
    return OpenCVTracker(init_frame, roi, tracker_type="KCF")

def create_csrt_tracker(init_frame, roi):
    return OpenCVTracker(init_frame, roi, tracker_type="CSRT")

def create_orb_tracker(init_frame, roi):
    return ORBTracker(init_frame, roi)

def create_bitplane_ens_tracker(init_frame, roi):
    return BitplaneEnsembleTracker(init_frame, roi)


# Trackers à tester
trackers_to_test = [
    ("TemplateMatch", create_template_tracker),
    ("BitPlane", create_bitplane_tracker),
    ("BitPlaneHist", create_bitplane_hist_tracker),
    ("BitplaneEns", create_bitplane_ens_tracker),
    ("Homography", create_homography_tracker),
    ("MOSSE", create_mosse_tracker),
    ("KCF", create_kcf_tracker),
    ("CSRT", create_csrt_tracker),
    ("ORB", create_orb_tracker),
]

# ------------------------------- Benchmark
def benchmark_video(args):
    video_path, gt_bboxes, method_name, tracker_factory = args

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or not gt_bboxes:
        return None

    first_frame_idx, roi = None, None
    for idx in sorted(gt_bboxes.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        x, y, w, h = map(int, gt_bboxes[idx])
        H, W = frame.shape[:2]
        if w > 0 and h > 0 and x >= 0 and y >= 0 and x+w <= W and y+h <= H:
            first_frame_idx = idx
            roi = (x, y, w, h)
            break

    if roi is None:
        cap.release()
        return None

    tracker = tracker_factory(frame, roi)
    tracked_bboxes = []
    ground_truth_bboxes = []
    fps_list = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    frame_id = first_frame_idx
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= MAX_FRAMES:
            break

        start = time.time()
        bbox, score = tracker.update(frame)
        fps_list.append(1.0 / max(time.time() - start, 1e-6))

        tracked_bboxes.append(bbox)
        ground_truth_bboxes.append(gt_bboxes.get(frame_id, None))

        if SHOW_PREVIEW:
            if frame_id in gt_bboxes:
                xg, yg, wg, hg = map(int, gt_bboxes[frame_id])
                cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
                cv2.putText(frame, "Ground Truth", (xg, yg - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if bbox:
                xt, yt, wt, ht = map(int, bbox)
                cv2.rectangle(frame, (xt, yt), (xt + wt, yt + ht), (0, 0, 255), 2)
                cv2.putText(frame, f"{method_name}", (xt, yt - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if frame_id in gt_bboxes:
                    iou = compute_iou(bbox, gt_bboxes[frame_id])
                    cv2.putText(frame, f"IoU {iou:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            frame_id += 1
            frame_count += 1
            cv2.imshow("Benchmark Visual", frame)
            if cv2.waitKey(1) == 27:  # ESC pour quitter
                break
    cap.release()

    avg_fps = np.mean(fps_list) if fps_list else 0
    acc = compute_accuracy(tracked_bboxes, ground_truth_bboxes)

    return {
        "method": method_name,
        "video": os.path.basename(video_path),
        "fps": avg_fps,
        "latency_ms": 1000.0 / max(avg_fps, 1e-6),
        "accuracy": acc
    }

# -------------------------------
# Callback pour écriture CSV par batch
# -------------------------------
def save_result_batch(buffer, output_csv, lock):
    """Écrit le batch dans le CSV et vide le buffer"""
    if not buffer:
        return

    df = pd.DataFrame(buffer)

    with lock:  # éviter collision entre process
        file_exists = os.path.exists(output_csv)
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)

    buffer.clear()

# -------------------------------
# Lancer le benchmark
# -------------------------------
if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)  # reset CSV

    loader = YTBBLoader(csv_path, videos_folder)
    all_tasks = []
    for method_name, tracker_factory in trackers_to_test:
        for video_path, gt_bboxes in loader.iter_video_segments(paths_only=True):
            all_tasks.append((video_path, gt_bboxes, method_name, tracker_factory))

    print(f"[INFO] {len(all_tasks)} tâches à traiter ({len(trackers_to_test)} méthodes * vidéos)")

    buffer = []         # local batch buffer
    batch_size = 5      # write every 5 results

    with Pool(processes=N_PROCESSES) as pool:
        for result in pool.imap_unordered(benchmark_video, all_tasks):
            buffer.append(result)

            if len(buffer) >= batch_size:
                df = pd.DataFrame(buffer)
                file_exists = os.path.exists(OUTPUT_CSV)
                df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
                buffer.clear()  # reset batch

    # flush remaining results
    if buffer:
        df = pd.DataFrame(buffer)
        file_exists = os.path.exists(OUTPUT_CSV)
        df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)

    print(f"[INFO] Benchmark terminé. Résultats dans {OUTPUT_CSV}")
