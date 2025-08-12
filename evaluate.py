import cv2
import numpy as np
import time
import csv
from collections import deque

# -------------------------------Chronométrage & FPS

class FPSCounter:
    def __init__(self, buffer_size=30):
        self.buffer = deque(maxlen=buffer_size)
        self.last_time = None

    def update(self):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            return None
        dt = now - self.last_time
        self.last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        self.buffer.append(fps)
        return self.average_fps()

    def average_fps(self):
        return np.mean(self.buffer) if self.buffer else 0.0

# -------------------------------Compute de IoU

def compute_iou(boxA, boxB):
    """
    IoU between two boxes : (x,y,w,h)
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return iou

# -------------------------------accuracy based on IoU

def compute_accuracy(tracked_bboxes, ground_truth_bboxes, iou_threshold=0.5):
    correct = 0
    total = 0
    for track, gt in zip(tracked_bboxes, ground_truth_bboxes):

        total += 1
        if track is not None and compute_iou(track, gt) >= iou_threshold:
            correct += 1
    return np.mean([compute_iou(t, g) for t, g in zip(tracked_bboxes, ground_truth_bboxes) if g is not None])


# -------------------------------Gestion des résultats et logs

def save_results_to_csv(filename, results, fieldnames):
    """
    results: dict
    fieldnames: columns
    """
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def print_results_table(results):
    """
    results: dict ; keys : method, fps, latency, accuracy
    """
    print("\n=== Benchmark Results ===")
    print(f"{'Method':15s} | {'FPS':>6} | {'Latency (ms)':>12} | {'Accuracy':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['method']:15s} | {r['fps']:6.2f} | {r['latency_ms']:12.2f} | {r['accuracy']*100:7.2f}%")
    print("-" * 50)
