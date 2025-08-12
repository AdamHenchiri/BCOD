import pandas as pd

csv_path = r"C:\Users\adamh\Documents\2025Internship\BCOD\youtube_boundingboxes_detection_validation.csv"
colnames = [
    "youtube_id", "timestamp_ms", "class_id", "class_name",
    "object_id", "object_presence", "xmin", "xmax", "ymin", "ymax"
]
df = pd.read_csv(csv_path, header=None, names=colnames)

print("Nombre de vidéos uniques :", df['youtube_id'].nunique())

unique_videos = df['youtube_id'].drop_duplicates()
n = min(10, len(unique_videos))
sample_videos = unique_videos.sample(n, random_state=0)

df_small = df[df['youtube_id'].isin(sample_videos)]

df_small.to_csv("ytbb_subset.csv", index=False)
print("Mini dataset of ", df_small['youtube_id'].nunique(), "videos", len(df_small), "lignes")
