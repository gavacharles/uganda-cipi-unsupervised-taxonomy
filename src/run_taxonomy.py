from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.optimize import linear_sum_assignment

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


@dataclass
class GridResult:
    normalization: str
    radius: int
    k: int
    silhouette: float
    gap: float
    gap_se: float


MACRO_LOWER = {
    "activity_indicator",
    "central_bank_rate",
    "cpi",
    "exchange_rate",
    "lending_rate",
    "ppi_manufacturing",
    "private_credit",
}


def zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def get_material_columns(panel: pd.DataFrame) -> List[str]:
    cols = []
    for c in panel.columns:
        if c.startswith("CIPI") or c.lower() in MACRO_LOWER:
            continue
        if c.upper() == c:
            cols.append(c)
    return cols


def prepare_variants(panel: pd.DataFrame, materials: List[str]) -> Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]:
    lvl = panel[materials].apply(pd.to_numeric, errors="coerce")
    lvl = lvl.interpolate(limit_direction="both").ffill().bfill()
    log_lvl = np.log(lvl)

    levels = zscore_rows(log_lvl.T.values)

    ret = log_lvl.diff().dropna(how="all")
    ret = ret.fillna(0.0)
    returns = zscore_rows(ret.T.values)

    t = np.arange(log_lvl.shape[0])
    detrended_rows = []
    for col in log_lvl.columns:
        y = log_lvl[col].values
        coef = np.polyfit(t, y, deg=1)
        trend = np.polyval(coef, t)
        detrended_rows.append(y - trend)
    detrended = zscore_rows(np.vstack(detrended_rows))

    return {
        "returns_z": (returns, ret.index),
        "levels_z": (levels, log_lvl.index),
        "detrended_levels_z": (detrended, log_lvl.index),
    }


def as_tslearn_shape(X: np.ndarray) -> np.ndarray:
    return X[:, :, None]


def dtw_distance_matrix(X: np.ndarray, radius: int) -> np.ndarray:
    return cdist_dtw(
        as_tslearn_shape(X),
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius,
    )


def fit_dba_kmeans(X: np.ndarray, k: int, radius: int, seed: int = RANDOM_SEED) -> TimeSeriesKMeans:
    return TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": radius},
        random_state=seed,
        n_init=4,
        max_iter=25,
        max_iter_barycenter=25,
        verbose=False,
    ).fit(as_tslearn_shape(X))


def gap_statistic(
    X: np.ndarray,
    k: int,
    radius: int,
    n_boot: int = 8,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed + 1000 + k + radius)
    km_obs = fit_dba_kmeans(X, k=k, radius=radius, seed=seed)
    w_obs = max(km_obs.inertia_, 1e-12)

    w_refs = []
    for _ in range(n_boot):
        X_ref = np.vstack([rng.permutation(row) for row in X])
        km_ref = fit_dba_kmeans(X_ref, k=k, radius=radius, seed=int(rng.integers(0, 10_000)))
        w_refs.append(max(km_ref.inertia_, 1e-12))

    log_refs = np.log(np.array(w_refs))
    gap = float(log_refs.mean() - np.log(w_obs))
    se = float(np.sqrt(1.0 + 1.0 / n_boot) * log_refs.std(ddof=1))
    return gap, se


def k_selection_grid(
    variants: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]],
    radii: Iterable[int],
    k_values: Iterable[int],
) -> pd.DataFrame:
    rows: List[GridResult] = []
    for norm_name, (X, _) in variants.items():
        for radius in radii:
            D = dtw_distance_matrix(X, radius)
            for k in k_values:
                km = fit_dba_kmeans(X, k=k, radius=radius)
                sil = silhouette_score(D, km.labels_, metric="precomputed")
                gap, gap_se = gap_statistic(X, k=k, radius=radius)
                rows.append(
                    GridResult(
                        normalization=norm_name,
                        radius=radius,
                        k=k,
                        silhouette=float(sil),
                        gap=float(gap),
                        gap_se=float(gap_se),
                    )
                )
    return pd.DataFrame([r.__dict__ for r in rows])


def pick_joint_optimum(grid: pd.DataFrame) -> pd.Series:
    g = grid.copy()
    g["sil_rank"] = g.groupby("normalization")["silhouette"].rank(method="min", ascending=False)
    g["gap_rank"] = g.groupby("normalization")["gap"].rank(method="min", ascending=False)
    g["joint_rank"] = g["sil_rank"] + g["gap_rank"]
    best = g.sort_values(["joint_rank", "silhouette", "gap"], ascending=[True, False, False]).iloc[0]
    return best


def pam_kmedoids(distance: np.ndarray, k: int, seed: int = RANDOM_SEED, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = distance.shape[0]
    medoids = np.array(rng.choice(n, size=k, replace=False))

    def assign(meds: np.ndarray) -> np.ndarray:
        return np.argmin(distance[:, meds], axis=1)

    labels = assign(medoids)
    best_cost = float(np.sum(distance[np.arange(n), medoids[labels]]))

    for _ in range(max_iter):
        improved = False
        for mi in range(k):
            for cand in range(n):
                if cand in medoids:
                    continue
                trial = medoids.copy()
                trial[mi] = cand
                trial_labels = assign(trial)
                trial_cost = float(np.sum(distance[np.arange(n), trial[trial_labels]]))
                if trial_cost + 1e-9 < best_cost:
                    medoids = trial
                    labels = trial_labels
                    best_cost = trial_cost
                    improved = True
        if not improved:
            break
    return labels


def ward_on_dtw(distance: np.ndarray, k: int, seed: int = RANDOM_SEED) -> np.ndarray:
    emb = MDS(
        n_components=min(8, distance.shape[0] - 1),
        dissimilarity="precomputed",
        random_state=seed,
        normalized_stress="auto",
    ).fit_transform(distance)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(emb)


def coassignment(labels: np.ndarray) -> np.ndarray:
    return (labels[:, None] == labels[None, :]).astype(float)


def mapped_labels(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    ref_u = np.unique(reference)
    tar_u = np.unique(target)
    mat = np.zeros((len(ref_u), len(tar_u)), dtype=int)
    for i, r in enumerate(ref_u):
        for j, t in enumerate(tar_u):
            mat[i, j] = int(np.sum((reference == r) & (target == t)))
    ridx, cidx = linear_sum_assignment(mat.max() - mat)
    mapping = {tar_u[c]: ref_u[r] for r, c in zip(ridx, cidx)}
    return np.array([mapping.get(v, v) for v in target])


def tsne_perplexity_sweep(distance: np.ndarray, perplexities: Iterable[int], seed: int = RANDOM_SEED) -> pd.DataFrame:
    rows = []
    n = distance.shape[0]
    for p in perplexities:
        peff = min(max(2, p), n - 1)
        emb = TSNE(
            n_components=2,
            metric="precomputed",
            random_state=seed,
            init="random",
            perplexity=peff,
            learning_rate="auto",
        ).fit_transform(distance)
        for i, (x, y) in enumerate(emb):
            rows.append({"idx": i, "method": "tsne", "perplexity": peff, "x": x, "y": y})

    try:
        import umap

        um = umap.UMAP(
            metric="precomputed",
            n_neighbors=min(10, n - 1),
            min_dist=0.1,
            random_state=seed,
        ).fit_transform(distance)
        for i, (x, y) in enumerate(um):
            rows.append({"idx": i, "method": "umap", "perplexity": np.nan, "x": x, "y": y})
    except Exception:
        pass
    return pd.DataFrame(rows)


def plot_heatmap(distance: np.ndarray, labels: np.ndarray, materials: List[str], out_path: Path) -> None:
    order = np.lexsort((np.array(materials), labels))
    D = distance[np.ix_(order, order)]
    sim = 1.0 / (1.0 + D)
    names = [materials[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="viridis", aspect="auto")
    ax.set_title("Cluster Heatmap: DTW Co-movement Intensity")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="1 / (1 + DTW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_manifold(emb: pd.DataFrame, labels: np.ndarray, materials: List[str], out_main: Path, out_appendix: Path) -> None:
    tsne = emb[(emb["method"] == "tsne") & (emb["perplexity"] == emb["perplexity"].min())].copy()
    tsne = tsne.sort_values("idx")

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tsne["x"], tsne["y"], c=labels, cmap="tab10", s=80, alpha=0.9)
    for _, r in tsne.iterrows():
        ax.text(r["x"], r["y"], materials[int(r["idx"])], fontsize=8)
    ax.set_title("t-SNE Manifold of DTW Distance Matrix")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(sc, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(out_main, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, p in enumerate(sorted(tsne["perplexity"].unique())):
        e = emb[(emb["method"] == "tsne") & (emb["perplexity"] == p)].sort_values("idx")
        axes[0].scatter(e["x"], e["y"], c=labels, cmap="tab10", s=60, alpha=0.9)
        axes[0].set_title("t-SNE perplexity sweep")
    um = emb[emb["method"] == "umap"].sort_values("idx")
    if not um.empty:
        axes[1].scatter(um["x"], um["y"], c=labels, cmap="tab10", s=60, alpha=0.9)
        axes[1].set_title("UMAP (precomputed DTW)")
    else:
        axes[1].set_title("UMAP unavailable")
    for ax in axes:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(out_appendix, dpi=180)
    plt.close(fig)


def plot_centroids(
    X: np.ndarray,
    time_index: pd.DatetimeIndex,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    k = len(np.unique(labels))
    fig, axes = plt.subplots(k, 1, figsize=(10, 2.4 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for cl in sorted(np.unique(labels)):
        ax = axes[cl]
        centroid = X[labels == cl].mean(axis=0)
        ax.plot(time_index, centroid, lw=1.8)
        ax.axvline(pd.Timestamp("2020-03-01"), color="firebrick", ls="--", alpha=0.7)
        ax.axvline(pd.Timestamp("2022-03-01"), color="darkorange", ls="--", alpha=0.7)
        ax.set_title(f"Cluster {cl} centroid profile")
        ax.grid(alpha=0.25)
    fig.suptitle("Centroid Profiles (z-scored log levels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_optimal_partition_and_divergence(
    grid: pd.DataFrame,
    divergence_df: pd.DataFrame,
    selected_norm: str,
    selected_radius: int,
    selected_k: int,
    out_path: Path,
) -> None:
    g = grid[(grid["normalization"] == selected_norm) & (grid["radius"] == selected_radius)].sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(g["k"], g["silhouette"], marker="o", lw=1.8, label="Silhouette")
    ax2 = ax.twinx()
    ax2.plot(g["k"], g["gap"], marker="s", lw=1.8, color="darkorange", label="Gap")
    ax.axvline(selected_k, color="firebrick", ls="--", alpha=0.8, label=f"Selected k={selected_k}")
    ax.set_title(f"Optimal partition (norm={selected_norm}, radius={selected_radius})")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax2.set_ylabel("Gap statistic")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    ax = axes[1]
    divm = divergence_df.melt(
        id_vars=["normalization"],
        value_vars=["ari_trade", "ari_isic_proxy"],
        var_name="benchmark",
        value_name="ari",
    )
    x = np.arange(divm["normalization"].nunique())
    norms = list(divm["normalization"].drop_duplicates())
    w = 0.36
    trade_vals = [float(divergence_df.loc[divergence_df["normalization"] == n, "ari_trade"].iloc[0]) for n in norms]
    isic_vals = [float(divergence_df.loc[divergence_df["normalization"] == n, "ari_isic_proxy"].iloc[0]) for n in norms]
    ax.bar(x - w / 2, trade_vals, width=w, label="ARI vs trade")
    ax.bar(x + w / 2, isic_vals, width=w, label="ARI vs ISIC proxy")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(norms, rotation=20)
    ax.set_ylim(min(-0.05, min(trade_vals + isic_vals) - 0.02), max(trade_vals + isic_vals) + 0.08)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("Taxonomy divergence by normalization")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_regime_structure_cross_trade(cluster_table: pd.DataFrame, out_path: Path) -> None:
    ctab = pd.crosstab(cluster_table["cluster"], cluster_table["trade_group"]).sort_index()
    row_share = ctab.div(ctab.sum(axis=1), axis=0).fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i, trade in enumerate(row_share.columns):
        left = row_share.iloc[:, :i].sum(axis=1).values if i > 0 else np.zeros(row_share.shape[0])
        ax.barh(row_share.index.astype(str), row_share[trade].values, left=left, label=trade)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Within-cluster trade composition share")
    ax.set_ylabel("Cluster")
    ax.set_title("Regime structure: cross-trade composition")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    im = ax.imshow(ctab.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ctab.columns)))
    ax.set_yticks(range(len(ctab.index)))
    ax.set_xticklabels(ctab.columns, rotation=45, ha="right")
    ax.set_yticklabels(ctab.index)
    ax.set_xlabel("Trade group")
    ax.set_ylabel("Cluster")
    ax.set_title("Cluster × trade count matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_estimator_triangulation(
    labels_kmeans: np.ndarray,
    labels_kmedoids: np.ndarray,
    labels_ward: np.ndarray,
    co_avg: np.ndarray,
    materials: List[str],
    out_path: Path,
) -> None:
    names = ["DBA k-means", "k-medoids", "Ward-on-MDS"]
    lbls = [labels_kmeans, labels_kmedoids, labels_ward]
    ari_mat = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            ari_mat[i, j] = adjusted_rand_score(lbls[i], lbls[j])
            ari_mat[j, i] = ari_mat[i, j]

    order = np.argsort(labels_kmeans)
    co_ord = co_avg[np.ix_(order, order)]
    mats_ord = [materials[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(ari_mat, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Estimator triangulation (pairwise ARI)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{ari_mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im2 = ax.imshow(co_ord, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(mats_ord)))
    ax.set_yticks(range(len(mats_ord)))
    ax.set_xticklabels(mats_ord, rotation=90)
    ax.set_yticklabels(mats_ord)
    ax.set_title("Average co-assignment across estimators")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Pr(same cluster)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temporal_stability(migration: pd.DataFrame, stability_ari: float, out_path: Path) -> None:
    trans = pd.crosstab(migration["cluster_first_half"], migration["cluster_second_half"]).sort_index(axis=0).sort_index(axis=1)
    rates = migration.groupby("cluster_first_half")["migrated"].mean().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(trans.values, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(trans.columns)))
    ax.set_yticks(range(len(trans.index)))
    ax.set_xticklabels(trans.columns)
    ax.set_yticklabels(trans.index)
    ax.set_xlabel("Second-half cluster")
    ax.set_ylabel("First-half cluster")
    ax.set_title(f"Cluster migration matrix (ARI={stability_ari:.3f})")
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            ax.text(j, i, str(trans.iloc[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    ax = axes[1]
    ax.bar(rates.index.astype(str), rates.values)
    ax.set_ylim(0, 1)
    ax.set_xlabel("First-half cluster")
    ax.set_ylabel("Migration share")
    ax.set_title("Within-cluster migration rates")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=str,
        default="/Users/charlesgava/Documents/Projects /Paper 3- Unsupervised Taxonomy/cio_pipeline-2/data/processed/panel_v1.0.parquet",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=str(Path(__file__).resolve().parent / "trade_mapping.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    if isinstance(panel.index, pd.DatetimeIndex):
        panel = panel.sort_index()

    materials = get_material_columns(panel)
    variants = prepare_variants(panel, materials)

    grid = k_selection_grid(variants, radii=[3, 6, 12], k_values=[3, 4, 5])
    grid.to_csv(outdir / "tables" / "k_selection_grid.csv", index=False)

    best = pick_joint_optimum(grid)
    selected_norm = str(best["normalization"])
    selected_radius = int(best["radius"])
    selected_k = int(best["k"])

    X_main, t_main = variants[selected_norm]
    D_main = dtw_distance_matrix(X_main, selected_radius)

    km = fit_dba_kmeans(X_main, k=selected_k, radius=selected_radius)
    labels_kmeans = km.labels_.astype(int)

    labels_kmedoids = pam_kmedoids(D_main, k=selected_k)
    labels_ward = ward_on_dtw(D_main, k=selected_k)

    co_kmeans = coassignment(labels_kmeans)
    co_kmedoids = coassignment(labels_kmedoids)
    co_ward = coassignment(labels_ward)
    co_avg = (co_kmeans + co_kmedoids + co_ward) / 3.0

    pd.DataFrame(co_avg, index=materials, columns=materials).to_csv(
        outdir / "tables" / "coassignment_average.csv"
    )

    agreement = pd.DataFrame(
        {
            "comparison": ["kmeans_vs_kmedoids", "kmeans_vs_ward", "kmedoids_vs_ward"],
            "adjusted_rand": [
                adjusted_rand_score(labels_kmeans, labels_kmedoids),
                adjusted_rand_score(labels_kmeans, labels_ward),
                adjusted_rand_score(labels_kmedoids, labels_ward),
            ],
        }
    )
    agreement.to_csv(outdir / "tables" / "estimator_agreement.csv", index=False)

    mapping = pd.read_csv(args.mapping)
    mapping = mapping.set_index("material").reindex(materials)

    ari_trade = adjusted_rand_score(labels_kmeans, mapping["trade_group"].astype("category").cat.codes)
    ari_isic = adjusted_rand_score(labels_kmeans, mapping["isic_proxy"].astype("category").cat.codes)

    X_levels, t_levels = variants["levels_z"]
    D_levels = dtw_distance_matrix(X_levels, selected_radius)
    km_levels = fit_dba_kmeans(X_levels, k=selected_k, radius=selected_radius)

    split = X_main.shape[1] // 2
    X_a, X_b = X_main[:, :split], X_main[:, split:]
    km_a = fit_dba_kmeans(X_a, k=selected_k, radius=min(selected_radius, max(2, X_a.shape[1] // 8)))
    km_b = fit_dba_kmeans(X_b, k=selected_k, radius=min(selected_radius, max(2, X_b.shape[1] // 8)))
    labels_b_map = mapped_labels(km_a.labels_, km_b.labels_)
    stability_ari = adjusted_rand_score(km_a.labels_, labels_b_map)

    migration = pd.DataFrame(
        {
            "material": materials,
            "cluster_first_half": km_a.labels_,
            "cluster_second_half": labels_b_map,
        }
    )
    migration["migrated"] = migration["cluster_first_half"] != migration["cluster_second_half"]
    migration.to_csv(outdir / "tables" / "cluster_migration.csv", index=False)

    cluster_table = pd.DataFrame(
        {
            "material": materials,
            "cluster": labels_kmeans,
            "trade_group": mapping["trade_group"].values,
            "isic_proxy": mapping["isic_proxy"].values,
            "supply_exposure": mapping["supply_exposure"].values,
        }
    ).sort_values(["cluster", "material"])
    cluster_table.to_csv(outdir / "tables" / "clusters_with_trade_mapping.csv", index=False)

    emb = tsne_perplexity_sweep(D_main, perplexities=[4, 6, 8, 10, 12])
    emb["material"] = emb["idx"].map({i: m for i, m in enumerate(materials)})
    emb["cluster"] = emb["idx"].map({i: int(c) for i, c in enumerate(labels_kmeans)})
    emb.to_csv(outdir / "tables" / "embedding_sweep.csv", index=False)

    plot_heatmap(D_main, labels_kmeans, materials, outdir / "figures" / "cluster_heatmap.png")
    plot_manifold(
        emb,
        labels_kmeans,
        materials,
        out_main=outdir / "figures" / "tsne_manifold.png",
        out_appendix=outdir / "figures" / "tsne_umap_appendix.png",
    )
    plot_centroids(X_levels, t_levels, km_levels.labels_.astype(int), outdir / "figures" / "centroid_profiles.png")

    norm_sensitivity = []
    divergence_rows = []
    for nm, (Xn, _) in variants.items():
        Dn = dtw_distance_matrix(Xn, selected_radius)
        lbn = fit_dba_kmeans(Xn, k=selected_k, radius=selected_radius).labels_
        norm_sensitivity.append(
            {
                "normalization": nm,
                "ari_vs_selected": adjusted_rand_score(labels_kmeans, lbn),
                "silhouette": silhouette_score(Dn, lbn, metric="precomputed"),
            }
        )
        divergence_rows.append(
            {
                "normalization": nm,
                "ari_trade": adjusted_rand_score(lbn, mapping["trade_group"].astype("category").cat.codes),
                "ari_isic_proxy": adjusted_rand_score(lbn, mapping["isic_proxy"].astype("category").cat.codes),
            }
        )
    pd.DataFrame(norm_sensitivity).to_csv(outdir / "tables" / "normalization_sensitivity.csv", index=False)
    divergence_df = pd.DataFrame(divergence_rows)
    divergence_df.to_csv(outdir / "tables" / "taxonomy_divergence_by_normalization.csv", index=False)

    euclid_sil = []
    for k in [3, 4, 5]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score as sil
        from sklearn.metrics.pairwise import euclidean_distances

        lbl = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20).fit_predict(X_main)
        De = euclidean_distances(X_main)
        euclid_sil.append({"k": k, "silhouette_euclidean": sil(De, lbl, metric="precomputed")})
    pd.DataFrame(euclid_sil).to_csv(outdir / "tables" / "euclidean_baseline.csv", index=False)

    plot_optimal_partition_and_divergence(
        grid=grid,
        divergence_df=divergence_df,
        selected_norm=selected_norm,
        selected_radius=selected_radius,
        selected_k=selected_k,
        out_path=outdir / "figures" / "optimal_partition_and_divergence.png",
    )
    plot_regime_structure_cross_trade(
        cluster_table=cluster_table,
        out_path=outdir / "figures" / "regime_structure_cross_trade.png",
    )
    plot_estimator_triangulation(
        labels_kmeans=labels_kmeans,
        labels_kmedoids=labels_kmedoids,
        labels_ward=labels_ward,
        co_avg=co_avg,
        materials=materials,
        out_path=outdir / "figures" / "estimator_triangulation.png",
    )
    plot_temporal_stability(
        migration=migration,
        stability_ari=stability_ari,
        out_path=outdir / "figures" / "temporal_stability.png",
    )

    manifest = {
        "seed": RANDOM_SEED,
        "panel_path": args.panel,
        "n_materials": len(materials),
        "materials": materials,
        "sample_start": str(panel.index.min().date()),
        "sample_end": str(panel.index.max().date()),
        "selected_normalization": selected_norm,
        "selected_dtw_radius": selected_radius,
        "selected_k": selected_k,
        "headline_ari_trade": float(ari_trade),
        "headline_ari_isic_proxy": float(ari_isic),
        "temporal_stability_ari": float(stability_ari),
        "migration_rate": float(migration["migrated"].mean()),
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
