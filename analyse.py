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
print("\n=== Top 10 par accuracy ===")
print(df_sorted.head(10))

# --- STATISTIQUES PAR MÉTHODE ---
stats = df.groupby("method").agg({
    "accuracy": ["mean", "max", "min"],
    "fps": "mean",
    "latency_ms": "mean"
}).reset_index()

print("\n=== Statistiques par méthode ===")
print(stats)

# --- GRAPHIQUE ACCURACY PAR MÉTHODE ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="accuracy")
plt.title("Accuracy moyenne par méthode")
plt.ylabel("Accuracy")
plt.xlabel("Méthode")
if save_plots:
    plt.savefig("graph/accuracy_par_methode.png")
plt.show()

# --- GRAPHIQUE LATENCE PAR MÉTHODE ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="latency_ms")
plt.title("Latence moyenne par méthode (ms)")
plt.ylabel("Latence (ms)")
plt.xlabel("Méthode")
if save_plots:
    plt.savefig("graph/latence_par_methode.png")
plt.show()

# --- GRAPHIQUE FPS PAR MÉTHODE ---
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="method", y="fps")
plt.title("FPS moyen par méthode")
plt.ylabel("FPS")
plt.xlabel("Méthode")
if save_plots:
    plt.savefig("graph/fps_par_methode.png")
plt.show()

# --- GRAPHIQUE ACCURACY vs LATENCE ---
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="latency_ms", y="accuracy", hue="method", style="method", s=80)
plt.title("Accuracy en fonction de la latence")
plt.xlabel("Latence (ms)")
plt.ylabel("Accuracy")
plt.grid(True)
if save_plots:
    plt.savefig("graph/accuracy_vs_latence.png")
plt.show()
