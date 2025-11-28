#!/usr/bin/env python3
"""
QC and visualization for nuclear receptor structure dataset.
Saves plots to PNGs instead of showing interactive windows.
"""

from pathlib import Path
from itertools import cycle
import pandas as pd
import numpy as np
import warnings

try:
    import umap
except Exception:  # pragma: no cover - optional dependency/runtime issues
    umap = None

try:
    import plotly.express as px
    from plotly import graph_objects as go
except Exception:  # pragma: no cover - optional dependency/runtime issues
    px = None
    go = None

# --- make matplotlib non-interactive & fast ---
import matplotlib
matplotlib.use("Agg")           # render to files, not GUI
import matplotlib.pyplot as plt

#ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


def _standardize_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z-scored numpy matrix plus mean/std for reference."""
    matrix = df[cols].to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std, mean, std


def _kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimal K-means implementation that avoids sklearn dependency."""
    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for init in range(n_init):
        if len(X) <= n_clusters:
            centers = X.copy()
        else:
            seed_idx = rng.choice(len(X), n_clusters, replace=False)
            centers = X[seed_idx].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            labels = dists.argmin(axis=1)

            new_centers = centers.copy()
            for cid in range(n_clusters):
                members = X[labels == cid]
                if len(members) == 0:
                    new_centers[cid] = X[rng.integers(len(X))]
                else:
                    new_centers[cid] = members.mean(axis=0)

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= tol:
                break

        inertia = np.sum((X - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.maximum(D, 0.0)
    return D


def _hbeta(dist_row: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    P = np.exp(-dist_row * beta)
    sumP = np.maximum(np.sum(P), 1e-12)
    H = np.log(sumP) + beta * np.sum(dist_row * P) / sumP
    return H, P / sumP


def _tsne_numpy(
    X: np.ndarray,
    *,
    perplexity: float = 30.0,
    n_components: int = 2,
    n_iter: int = 1000,
    learning_rate: float = 200.0,
    early_exaggeration: float = 4.0,
    random_state: int = 42,
) -> np.ndarray:
    """Small t-SNE implementation (exact algorithm, no Barnes-Hut)."""
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples.")

    distances = _pairwise_distances(X)
    P = np.zeros_like(distances)
    log_perp = np.log(perplexity)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        beta = 1.0
        betamin, betamax = -np.inf, np.inf
        Di = distances[i, mask]

        for _ in range(50):
            H, thisP = _hbeta(Di, beta)
            Hdiff = H - log_perp
            if np.abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax == np.inf else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin == -np.inf else (beta + betamin) / 2

        P[i, mask] = thisP

    P = (P + P.T)
    P /= np.sum(P)
    P = np.maximum(P, 1e-12)
    P *= early_exaggeration

    rng = np.random.default_rng(random_state)
    Y = rng.normal(0.0, 1e-4, size=(n_samples, n_components))
    gains = np.ones_like(Y)
    y_inertia = np.zeros_like(Y)

    for iteration in range(n_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = 1 / (1 + sum_Y[:, None] + sum_Y[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = (P - Q) * num
        dY = 4 * np.sum(PQ[:, :, None] * (Y[:, None, :] - Y[None, :, :]), axis=1)

        momentum = 0.5 if iteration < 250 else 0.8
        gains = (gains + 0.2) * (np.sign(dY) != np.sign(y_inertia)) + (gains * 0.8) * (
            np.sign(dY) == np.sign(y_inertia)
        )
        gains[gains < 0.01] = 0.01

        y_inertia = momentum * y_inertia - learning_rate * (gains * dY)
        Y += y_inertia
        Y -= Y.mean(axis=0, keepdims=True)

        if iteration == 250:
            P /= early_exaggeration

    return Y


def _save_interactive_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_path: Path,
    *,
    title: str,
    color_col: str = "cluster_label",
) -> None:
    """Persist Plotly scatter with hover metadata when plotly is available."""
    if px is None:
        print(f"[WARN] Plotly not installed; skipping interactive plot '{plot_path.name}'.")
        return

    hover_data = {
        "pdb_id": True,
        "symbol": True,
        "receptor_class": True,
        "subfamily": True,
        "resolution": ":.2f",
        "cluster_label": True,
        "mean_bfactor": ":.2f",
        "std_bfactor": ":.2f",
    }

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="pdb_id",
        hover_data=hover_data,
        title=title,
        width=900,
        height=700,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=0.05, color="#333333")))
    fig.update_layout(legend_title_text="Cluster", template="plotly_white")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_path))
    print(f"Saved interactive scatter → {plot_path}")


def _save_symbol_cluster_toggle_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_path: Path,
    *,
    title: str,
    class_col: str = "receptor_class",
) -> None:
    """Interactive scatter with receptor legend + buttons for coloring and class filters."""
    if px is None or go is None:
        print(f"[WARN] Plotly not installed; skipping interactive plot '{plot_path.name}'.")
        return

    if df.empty:
        print(f"[WARN] No data for interactive plot '{plot_path.name}'.")
        return

    df = df.copy()
    df = df.dropna(subset=[x_col, y_col, "symbol", "cluster_label"])
    if df.empty:
        print(f"[WARN] Missing coordinates or annotations for '{plot_path.name}'.")
        return

    if class_col not in df.columns:
        df[class_col] = DEFAULT_RECEPTOR_CLASS
    df[class_col] = df[class_col].fillna(DEFAULT_RECEPTOR_CLASS)

    symbol_palette = (
        px.colors.qualitative.Safe
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Bold
    )
    cluster_palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.Dark24
    )

    symbol_cycle = cycle(symbol_palette)
    cluster_cycle = cycle(cluster_palette)
    symbol_colors: dict[str, str] = {}
    cluster_colors: dict[str, str] = {}

    symbols = sorted(df["symbol"].dropna().unique())
    clusters = sorted(df["cluster_label"].dropna().unique())

    for sym in symbols:
        symbol_colors[sym] = next(symbol_cycle)
    for cid in clusters:
        cluster_colors[cid] = next(cluster_cycle)

    fig = go.Figure()
    hover_template = (
        "PDB: %{customdata[0]}<br>"
        "Receptor: %{customdata[1]}<br>"
        "Class: %{customdata[2]}<br>"
        "Cluster: %{customdata[3]}<br>"
        "Resolution: %{customdata[4]:.2f} Å<br>"
        "Mean B: %{customdata[5]:.2f}<br>"
        "Std B: %{customdata[6]:.2f}<extra></extra>"
    )

    receptor_color_arrays: list[list[str]] = []
    cluster_color_arrays: list[list[str]] = []
    trace_classes: list[str] = []

    for sym in symbols:
        sub = df[df["symbol"] == sym]
        if sub.empty:
            continue
        receptor_class = sub[class_col].iloc[0]
        trace_classes.append(receptor_class)
        receptor_color_list = [symbol_colors[sym]] * len(sub)
        cluster_color_list = [cluster_colors[c] for c in sub["cluster_label"]]
        receptor_color_arrays.append(receptor_color_list)
        cluster_color_arrays.append(cluster_color_list)
        customdata = np.column_stack(
            [
                sub["pdb_id"].astype(str),
                sub["symbol"].astype(str),
                sub[class_col].astype(str),
                sub["cluster_label"].astype(str),
                sub["resolution"].astype(float),
                sub["mean_bfactor"].astype(float),
                sub["std_bfactor"].astype(float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=sym,
                legendgroup=sym,
                marker=dict(
                    color=receptor_color_list,
                    size=10,
                    opacity=0.85,
                    line=dict(width=0.4, color="#222222"),
                ),
                customdata=customdata,
                hovertemplate=hover_template,
                visible=True,
            )
        )

    if not fig.data:
        print(f"[WARN] No traces generated for '{plot_path.name}'.")
        return

    receptor_button = dict(
        label="Color by receptor",
        method="restyle",
        args=[{"marker.color": receptor_color_arrays}],
    )
    cluster_button = dict(
        label="Color by cluster",
        method="restyle",
        args=[{"marker.color": cluster_color_arrays}],
    )

    class_masks = {}
    total_traces = len(fig.data)
    class_masks["All"] = [True] * total_traces
    unique_classes = [cls for cls in RECEPTOR_CLASS_ORDER if cls != "All"]
    for cls in unique_classes:
        mask = [trace_cls == cls for trace_cls in trace_classes]
        class_masks[cls] = mask

    class_buttons = []
    for cls in RECEPTOR_CLASS_ORDER:
        if cls == "All":
            mask = class_masks["All"]
        else:
            mask = class_masks.get(cls)
            if mask is None:
                continue
        class_buttons.append(
            dict(
                label=cls,
                method="update",
                args=[{"visible": mask}],
            )
        )

    fig.update_layout(
        title=title,
        width=1000,
        height=720,
        template="plotly_white",
        legend_title_text="Receptor",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[receptor_button, cluster_button],
                x=1.18,
                xanchor="left",
                y=1.08,
                yanchor="top",
                pad={"r": 10, "t": 10},
                showactive=True,
            ),
            dict(
                type="buttons",
                direction="down",
                buttons=class_buttons,
                x=1.18,
                xanchor="left",
                y=0.82,
                pad={"r": 8, "t": 5},
                showactive=True,
            ),
        ],
        margin=dict(l=60, r=10, t=90, b=60),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_path))
    print(f"Saved interactive scatter → {plot_path}")


RECEPTOR_CLASS_MAP = {
    "NR3A": "Steroid (classic endocrine)",
    "NR3B": "Steroid (classic endocrine)",
    "NR3C": "Steroid (classic endocrine)",
    "NR1A": "Non-steroid endocrine",
    "NR1B": "Non-steroid endocrine",
    "NR5A": "Non-steroid endocrine",
    "NR1C": "Metabolic/xeno-sensors",
    "NR1H": "Metabolic/xeno-sensors",
    "NR1I": "Metabolic/xeno-sensors",
    "NR1D": "Circadian/immune modulators",
    "NR1F": "Circadian/immune modulators",
    "NR2F": "Circadian/immune modulators",
    "NR4A": "Circadian/immune modulators",
}
DEFAULT_RECEPTOR_CLASS = "Orphans"
RECEPTOR_CLASS_ORDER = [
    "All",
    "Steroid (classic endocrine)",
    "Non-steroid endocrine",
    "Metabolic/xeno-sensors",
    "Circadian/immune modulators",
    "Orphans",
]


# Paths
meta_path = Path("data/meta/structures.csv")
bf_path   = Path("data/processed/ca_bfactors.csv")
plot_dir  = Path("data/meta/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Load
meta = pd.read_csv(meta_path)
bf   = pd.read_csv(bf_path)

proteins = pd.read_csv("proteins.csv")

# Which proteins have at least one PDB in meta
covered = set(meta["uniprot"])
proteins["has_structure"] = proteins["uniprot"].isin(covered)

missing_proteins = proteins[~proteins["has_structure"]]

print("\nProteins with no structures:")
print(missing_proteins)

# Some people named the column "b_factor" in older versions, normalize:
if "bfactor" not in bf.columns and "b_factor" in bf.columns:
    bf = bf.rename(columns={"b_factor": "bfactor"})

print("Loaded:")
print(f"  Structures meta: {len(meta)} rows")
print(f"  B-factor rows:   {len(bf)}")
print(f"  Unique PDBs in B-factors: {bf['pdb_id'].nunique()}")

# coverage
counts = meta.groupby("symbol")["pdb_id"].nunique().sort_values(ascending=False)
print("\nStructures per protein (top 15):")
print(counts.head(15))

missing = counts[counts == 0]
if not missing.empty:
    print("\nProteins with no PDB structures:")
    print(missing.index.tolist())

#resolution / R-free QC
# bad_res = meta[meta["resolution"] > 3.0]
# if len(bad_res) > 0:
#     print(f"\n{len(bad_res)} structures have resolution worse than 3.0 Å:")
#     print(bad_res[["pdb_id", "symbol", "resolution"]].to_string(index=False))
# else:
#     print("\nAll structures have acceptable resolution (≤3.5 Å)")

# if meta["rfree"].isna().any():
#     print(f"{meta['rfree'].isna().sum()} structures missing R-free.")

# plot - Resolution distribution per subfamily
plt.figure(figsize=(10,6))
meta.boxplot(column="resolution", by="subfamily", rot=45)
plt.ylabel("Resolution (Å)")
plt.title("Distribution of structure resolution per NR subfamily")
plt.suptitle("")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_by_subfamily.png", dpi=300)
plt.close()

#plot: Average B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].mean().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Average Cα B-factor")
plt.title("Average flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "avg_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot: Std B-factor per receptor
avg_b = bf.groupby("symbol")["bfactor"].std().sort_values()
plt.figure(figsize=(10,12))
avg_b.plot(kind="barh")
plt.xlabel("Std Cα B-factor")
plt.title("Std flexibility per receptor (all structures)")
plt.tight_layout()
plt.savefig(plot_dir / "std_bfactor_per_receptor.png", dpi=300)
plt.close()

#plot resolution vs mean B-factor per structure
merged = (bf.groupby("pdb_id")["bfactor"].mean()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(7,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean Cα B-factor")

a, b = np.polyfit(merged["resolution"], merged["bfactor"], 1)
x = np.array([merged["resolution"].min(), merged["resolution"].max()])
plt.plot(x, a*x + b, color="red", linestyle="--", label=f"y={a:.1f}x+{b:.1f}")
plt.legend()

merged_clean = merged.dropna(subset=["resolution", "bfactor"])
corr = merged_clean["resolution"].corr(merged_clean["bfactor"])
print(f"\nCorrelation between resolution and mean B-factor: {corr:.3f}")


plt.title("Resolution vs mean B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_bfactor.png", dpi=300)
plt.close()

#resolution vs std b factor
merged = (bf.groupby("pdb_id")["bfactor"].std()
            .reset_index()
            .merge(meta, on="pdb_id", how="left"))

plt.figure(figsize=(10,6))
plt.scatter(merged["resolution"], merged["bfactor"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Std Cα B-factor")
plt.title("Resolution vs std B-factor per structure")
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_std_bfactor.png", dpi=300)
plt.close()

#summary table
summary = (bf.groupby("pdb_id")["bfactor"]
             .agg(mean="mean", std="std", count="count")
             .merge(meta, on="pdb_id", how="left"))
summary_out = Path("data/meta/structure_summary.csv")
summary.to_csv(summary_out, index=False)

print("\nSummary saved:", summary_out)
print("Plots saved to:", plot_dir)
print("QC complete (non-interactive).")

# --- Per-receptor correlation ---
corr_per_receptor = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("symbol")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)
# --- Per-subfamily correlation ---
corr_per_receptor.to_csv("data/meta/correlation_per_receptor.csv", index=False)


corr_per_subfam = (
    summary.dropna(subset=["mean", "resolution"])
           .groupby("subfamily")
           .apply(lambda df: df["resolution"].corr(df["mean"]))
           .reset_index(name="correlation")
)

corr_per_subfam.to_csv("data/meta/correlation_per_subfamily.csv", index=False)

# --- Mean ± SD B-factors and all resolutions per receptor ---
b_summary = (
    summary.groupby("symbol")
           .agg(mean_bfactor=("mean", "mean"),
                std_bfactor=("mean", "std"),
                n_structures=("pdb_id", "count"))
           .reset_index()
)

res_lookup = (summary.groupby("symbol")["resolution"]
                     .apply(lambda x: ", ".join(sorted(map(lambda v: f"{v:.2f}", x.unique()))))
                     .reset_index(name="resolutions"))

b_summary = b_summary.merge(res_lookup, on="symbol", how="left")
b_summary.to_csv("data/meta/bfactor_resolution_summary_per_receptor.csv", index=False)

print("\nSaved: bfactor_resolution_summary_per_receptor.csv")


# Plot: Mean ± SD B-factors per receptor
b_summary = pd.read_csv("data/meta/bfactor_resolution_summary_per_receptor.csv")
b_summary = b_summary.sort_values("mean_bfactor", ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(b_summary["symbol"],
         b_summary["mean_bfactor"],
         xerr=b_summary["std_bfactor"],
         color="skyblue",
         ecolor="gray",
         capsize=3)

plt.xlabel("Mean Cα B-factor (Å²)")
plt.ylabel("Receptor (symbol)")
plt.title("Mean ± SD B-factors per receptor")
plt.tight_layout()
plt.savefig("data/meta/plots/mean_sd_bfactor_per_receptor.png", dpi=300)
plt.close()


# ============================
# normalization by resolution bins
# ============================

# attach resolution to every residue B-factor row
bf_with_res = bf.merge(meta[["pdb_id","resolution"]], on="pdb_id", how="left")
bf_with_res = bf_with_res.dropna(subset=["resolution", "bfactor"])

# define resolution bins
bin_edges  = [1.0, 2.0, 2.5, 3.0, 3.5, 4.5, 10.0]
bin_labels = ["1.0–2.0", "2.0–2.5", "2.5–3.0", "3.0–3.5", "3.5–4.5", "4.5–10.0"]
bf_with_res["res_bin"] = pd.cut(bf_with_res["resolution"], bins=bin_edges,
                                labels=bin_labels, include_lowest=True, right=True)

# bin stats
bin_stats = (bf_with_res.groupby("res_bin")["bfactor"]
             .agg(mean_bin="mean", std_bin="std").reset_index())

# if std_bin is 0 or NaN (very small bin), set a floor value
EPS = 1e-6
bin_stats["std_bin"] = bin_stats["std_bin"].fillna(0.0).clip(lower=EPS)

# merge stats back and compute z-score per residue: z = (B - mean_bin) / std_bin
bf_norm = bf_with_res.merge(bin_stats, on="res_bin", how="left")
bf_norm["z_bfactor"] = (bf_norm["bfactor"] - bf_norm["mean_bin"]) / bf_norm["std_bin"]

# save normalized residue-level table
norm_out = Path("data/processed/ca_bfactors_normalized.csv")
bf_norm.to_csv(norm_out, index=False)
print(f"\nWrote z-normalized residue table → {norm_out}")

# average z across its residues
summary_norm = (bf_norm.groupby("pdb_id")["z_bfactor"]
                .agg(mean_norm="mean", std_norm="std", count="count")
                .reset_index()
                .merge(meta, on="pdb_id", how="left"))
summary_norm_out = Path("data/meta/structure_summary_normalized.csv")
summary_norm.to_csv(summary_norm_out, index=False)
print(f"Wrote normalized per-structure summary → {summary_norm_out}")

# correlation check: before vs after normalization
raw_summary = pd.read_csv("data/meta/structure_summary.csv").dropna(subset=["mean","resolution"])
print("Correlation BEFORE normalization (resolution vs mean B):",
      f"{raw_summary['resolution'].corr(raw_summary['mean']):.3f}")

summary_norm_clean = summary_norm.dropna(subset=["mean_norm","resolution"])
print("Correlation AFTER normalization (resolution vs mean_norm):",
      f"{summary_norm_clean['resolution'].corr(summary_norm_clean['mean_norm']):.3f}")



#plot: Resolution vs mean_norm 
plt.figure(figsize=(7,6))
plt.scatter(summary_norm_clean["resolution"], summary_norm_clean["mean_norm"], alpha=0.6)
plt.xlabel("Resolution (Å)")
plt.ylabel("Mean normalized Cα B-factor (z)")
plt.title("Resolution vs mean normalized B-factor per structure")
#best-fit line
a2, b2 = np.polyfit(summary_norm_clean["resolution"], summary_norm_clean["mean_norm"], 1)
xx = np.array([summary_norm_clean["resolution"].min(), summary_norm_clean["resolution"].max()])
plt.plot(xx, a2*xx + b2, linestyle="--", label=f"y={a2:.2f}x+{b2:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "resolution_vs_mean_norm_bfactor.png", dpi=300)
plt.close()

# Plot: Mean normalized B-factor per receptor
per_receptor_norm = (summary_norm_clean.groupby("symbol")["mean_norm"]
                     .mean().sort_values())
plt.figure(figsize=(10,12))
per_receptor_norm.plot(kind="barh")
plt.xlabel("Mean normalized Cα B-factor (z)")
plt.title("Normalized flexibility per receptor (mean of structure z-means)")
plt.tight_layout()
plt.savefig(plot_dir / "mean_norm_bfactor_per_receptor.png", dpi=300)
plt.close()


# Per-receptor correlation
def _safe_corr(g):
    g = g.dropna(subset=["resolution","mean_norm"])
    return g["resolution"].corr(g["mean_norm"]) if len(g) > 2 else np.nan
corr_per_receptor_norm = (summary_norm.groupby("symbol", group_keys=False)
                          .apply(_safe_corr)
                          .reset_index(name="correlation_norm"))

corr_per_receptor_norm.to_csv("data/meta/correlation_per_receptor_NORMALIZED.csv", index=False)
print("\nSaved: correlation_per_receptor_NORMALIZED.csv")

# Per-subfamily correlation
corr_per_subfam_norm = (summary_norm.groupby("subfamily", group_keys=False)
                        .apply(_safe_corr)
                        .reset_index(name="correlation_norm"))
corr_per_subfam_norm.to_csv("data/meta/correlation_per_subfamily_NORMALIZED.csv", index=False)
print("Saved: correlation_per_subfamily_NORMALIZED.csv")

# ============================
# K-means clustering + t-SNE visualization (per structure)
# ============================

structure_stats = (
    bf.groupby("pdb_id")
      .agg(
          mean_bfactor=("bfactor", "mean"),
          std_bfactor=("bfactor", "std"),
          median_bfactor=("bfactor", "median"),
          min_bfactor=("bfactor", "min"),
          max_bfactor=("bfactor", "max"),
          q25=("bfactor", lambda s: s.quantile(0.25)),
          q75=("bfactor", lambda s: s.quantile(0.75)),
          residue_count=("bfactor", "count"),
          symbol=("symbol", "first"),
          subfamily=("subfamily", "first"),
          uniprot=("uniprot", "first"),
      )
      .reset_index()
      .merge(meta[["pdb_id", "resolution"]], on="pdb_id", how="left")
)
structure_stats["receptor_class"] = (
    structure_stats["subfamily"]
    .map(RECEPTOR_CLASS_MAP)
    .fillna(DEFAULT_RECEPTOR_CLASS)
)

feature_cols = [
    "mean_bfactor",
    "std_bfactor",
    "median_bfactor",
    "min_bfactor",
    "max_bfactor",
    "q25",
    "q75",
    "residue_count",
]

structure_stats[feature_cols] = structure_stats[feature_cols].fillna(0.0)

if len(structure_stats) < 2:
    print("\nNot enough structures for clustering; skipping K-means/t-SNE analysis.")
else:
    X_scaled, _, _ = _standardize_matrix(structure_stats, feature_cols)

    desired_clusters = 5
    best_k = min(desired_clusters, len(structure_stats))

    labels, centers, inertia = _kmeans_numpy(
        X_scaled,
        n_clusters=best_k,
        n_init=25,
        random_state=42,
    )
    structure_stats["cluster_id"] = labels
    structure_stats["cluster_label"] = structure_stats["cluster_id"].apply(lambda cid: f"C{cid + 1}")

    if len(structure_stats) < 3:
        print("\nNot enough structures for t-SNE projection; saved clusters without embedding.")
    else:
        perplexity = min(30, len(structure_stats) - 1)
        perplexity = max(2, perplexity)

        tsne_coords = _tsne_numpy(
            X_scaled,
            perplexity=perplexity,
            random_state=42,
        )

        structure_stats["tsne_1"] = tsne_coords[:, 0]
        structure_stats["tsne_2"] = tsne_coords[:, 1]

        plt.figure(figsize=(9, 7))
        cluster_cmap = plt.cm.get_cmap("tab10", best_k)
        for idx, (cid, cluster_df) in enumerate(structure_stats.groupby("cluster_label")):
            plt.scatter(
                cluster_df["tsne_1"],
                cluster_df["tsne_2"],
                color=cluster_cmap(idx),
                alpha=0.7,
                label=cid,
                s=25,
            )

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("Per-structure Cα B-factor profiles colored by K-means cluster")
        plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        tsne_plot_path = plot_dir / "tsne_kmeans_bfactor_clusters.png"
        plt.savefig(tsne_plot_path, dpi=300)
        plt.close()
        tsne_html_path = plot_dir / "tsne_kmeans_bfactor_clusters.html"
        _save_interactive_scatter(
            structure_stats,
            "tsne_1",
            "tsne_2",
            tsne_html_path,
            title="Per-structure Cα B-factor profiles (t-SNE, interactive)",
        )
        tsne_symbol_html = plot_dir / "tsne_kmeans_receptors.html"
        _save_symbol_cluster_toggle_scatter(
            structure_stats,
            "tsne_1",
            "tsne_2",
            tsne_symbol_html,
            title="Per-structure Cα B-factor profiles (t-SNE, receptor legend)",
        )

    if umap is None:
        print("\numap-learn not installed; skipping UMAP visualization.")
    else:
        if len(structure_stats) < 3:
            print("\nNot enough structures for UMAP projection; skipping UMAP plot.")
        else:
            n_neighbors = min(15, len(structure_stats) - 1)
            n_neighbors = max(2, n_neighbors)
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.15,
                n_components=2,
                metric="euclidean",
                random_state=42,
            )
            umap_coords = reducer.fit_transform(X_scaled)
            structure_stats["umap_1"] = umap_coords[:, 0]
            structure_stats["umap_2"] = umap_coords[:, 1]

            plt.figure(figsize=(9, 7))
            cluster_cmap = plt.cm.get_cmap("tab10", best_k)
            for idx, (cid, cluster_df) in enumerate(structure_stats.groupby("cluster_label")):
                plt.scatter(
                    cluster_df["umap_1"],
                    cluster_df["umap_2"],
                    color=cluster_cmap(idx),
                    alpha=0.7,
                    label=cid,
                    s=25,
                )

            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.title("Per-structure Cα B-factor profiles colored by K-means cluster (UMAP)")
            plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            umap_plot_path = plot_dir / "umap_kmeans_bfactor_clusters.png"
            plt.savefig(umap_plot_path, dpi=300)
            plt.close()
            umap_html_path = plot_dir / "umap_kmeans_bfactor_clusters.html"
            _save_interactive_scatter(
                structure_stats,
                "umap_1",
                "umap_2",
                umap_html_path,
                title="Per-structure Cα B-factor profiles (UMAP, interactive)",
            )
            umap_symbol_html = plot_dir / "umap_kmeans_receptors.html"
            _save_symbol_cluster_toggle_scatter(
                structure_stats,
                "umap_1",
                "umap_2",
                umap_symbol_html,
                title="Per-structure Cα B-factor profiles (UMAP, receptor legend)",
            )

    cluster_out = Path("data/meta/bfactor_clusters_tsne.csv")
    structure_stats.to_csv(cluster_out, index=False)

    print(f"\nSaved per-structure K-means clusters → {cluster_out}")
    if "tsne_plot_path" in locals():
        print("Saved t-SNE plot →", tsne_plot_path)
    if "umap_plot_path" in locals():
        print("Saved UMAP plot →", umap_plot_path)
