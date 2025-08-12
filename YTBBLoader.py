import cv2
import pandas as pd
import os


class YTBBLoader:
    def __init__(self, csv_path, videos_folder):
        self.df = pd.read_csv(csv_path)
        self.videos_folder = videos_folder

        # Convert
        self.df['bbox'] = self.df.apply(self._norm_to_bbox, axis=1)

    def _norm_to_bbox(self, row):
        return (row['xmin'], row['ymin'], row['xmax'], row['ymax'])

    def iter_video_segments(self, paths_only=False):
        grouped = self.df.groupby(['youtube_id', 'object_id'])
        for (yt_id, obj_id), group in grouped:
            video_path = os.path.join(self.videos_folder, yt_id + ".mp4")
            if not os.path.exists(video_path):
                print(f"[WARN] Video {video_path} not found, skipping")
                continue

            width = height = None
            if not paths_only:
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                cap_tmp = cv2.VideoCapture(video_path)
                width = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_tmp.release()

            # Convert bbox to pixels
            gt_bboxes = {}
            fps = 30.0  # fallback
            if not paths_only:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            else:
                cap_tmp = cv2.VideoCapture(video_path)
                fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
                cap_tmp.release()

            for _, row in group.iterrows():
                x0 = row['xmin'] * width
                x1 = row['xmax'] * width
                y0 = row['ymin'] * height
                y1 = row['ymax'] * height
                w = x1 - x0
                h = y1 - y0
                frame_idx = int(row['timestamp_ms'] / 1000.0 * fps)
                gt_bboxes[frame_idx] = (int(x0), int(y0), int(w), int(h))

            if paths_only:
                yield video_path, gt_bboxes
            else:
                yield cap, gt_bboxes
