import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")

# --- CONFIG
csv_file = "benchmark_results.csv"
save_plots = True

df = pd.read_csv(csv_file)

# --- RÉORDONNER PAR ACCURACY DESCENDANT ---
df_sorted = df.sort_values(by="accuracy", ascending=False)
print("\n=== Top 10 per accuracy ===")
print(df_sorted.head(10))

# --- STAT/method ---
stats = df.groupby("method").agg({
    "accuracy": ["mean", "max", "min"],
    "fps": "mean",
    "latency_ms": "mean"
}).reset_index()

print("\n=== Stat by methode ===")
print(stats)

# --- GRAPH ACCURACY FOR EACH METHOD ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="accuracy")
plt.title("Accuracy moyenne par méthode")
plt.ylabel("Accuracy")
plt.xlabel("Méthode")
if save_plots:
    plt.savefig("graph/accuracy_by_methode.png")
plt.show()

# --- GRAPH LATENCY FOR EACH METHOD ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="latency_ms")
plt.title("avg latency per methode (ms)")
plt.ylabel("Latency (ms)")
plt.xlabel("Methode")
if save_plots:
    plt.savefig("graph/latency_by_methode.png")
plt.show()

# --- GRAPH FPS FOR EACH METHOD ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="fps")
plt.title("avg FPS by méthode")
plt.ylabel("FPS")
plt.xlabel("Methode")
if save_plots:
    plt.savefig("graph/fps_by_methode.png")
plt.show()

# --- GRAPH ACCURACY vs LATENCY ---
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="latency_ms", y="accuracy", hue="method", style="method", s=80)
plt.title("Accuracy by latency")
plt.xlabel("latency (ms)")
plt.ylabel("Accuracy")
plt.grid(True)
if save_plots:
    plt.savefig("graph/accuracy_vs_latency.png")
plt.show()
