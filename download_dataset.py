import pandas as pd
import os
import yt_dlp

csv_path = "ytbb_subset.csv"
videos_folder = "yt_videos"
os.makedirs(videos_folder, exist_ok=True)

df = pd.read_csv(csv_path)
for yt_id in df['youtube_id'].drop_duplicates():
    url = f"https://youtu.be/{yt_id}"
    output_path = os.path.join(videos_folder, f"{yt_id}.mp4")

    if os.path.exists(output_path):
        print(f"[OK] {yt_id} already exists, skipping download")
        continue

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_path,
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"[OK] {yt_id} downloaded")
    except Exception as e:
        print(f"[FAIL] {yt_id} : {e}")
