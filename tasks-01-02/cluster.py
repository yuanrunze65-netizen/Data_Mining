"""Unsupervised image clustering followed by a separate label-based evaluation."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SEED = 2026
K = 6
NEIGHBOURS = (8, 12, 16, 24, 32)


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def scaled_block(values):
    return normalize(np.sqrt(np.maximum(values, 0)))


def kmeans(values, seed=SEED, restarts=30):
    return KMeans(n_clusters=K, n_init=restarts, random_state=seed).fit_predict(values)


def spectral_basis(distances, neighbours):
    ranking_distances = distances.copy()
    np.fill_diagonal(ranking_distances, np.inf)
    order = np.argsort(ranking_distances, axis=1, kind="stable")[:, :neighbours]
    rows = np.arange(len(distances))[:, None]
    scale = np.maximum(distances[rows[:, 0], order[:, -1]], 1e-6)
    weights = np.exp(-(distances[rows, order] ** 2) / (scale[:, None] * scale[order]))
    adjacency = np.zeros_like(distances)
    adjacency[rows, order] = weights
    adjacency = np.maximum(adjacency, adjacency.T)
    components = connected_components(adjacency, directed=False, return_labels=False)
    degree = adjacency.sum(axis=1) ** -0.5
    similarity = adjacency * degree[:, None] * degree[None, :]
    n = len(similarity)
    eig, vectors = eigh(similarity, subset_by_index=(n - K - 1, n - 1))
    return normalize(vectors[:, -K:]), components, eig[1] - eig[0]


def fit(path, out):
    data = np.load(path)
    blocks = {name: scaled_block(data[name]) for name in ("avg3", "avg4", "std2", "std3", "colour")}
    texture = normalize(StandardScaler().fit_transform(data["texture"]))
    handcrafted = np.hstack([np.sqrt(.8) * blocks["colour"], np.sqrt(.2) * texture])
    deep = np.hstack([np.sqrt(.4) * blocks["avg3"], np.sqrt(.6) * blocks["avg4"]])
    hybrid = np.hstack([
        np.sqrt(.80) * deep,
        np.sqrt(.08) * blocks["std2"],
        np.sqrt(.07) * blocks["std3"],
        np.sqrt(.05) * blocks["colour"],
    ])
    distances = euclidean_distances(hybrid).astype(np.float64)
    groups = {
        "Handcrafted K-means": kmeans(handcrafted),
        "Deep K-means": kmeans(deep),
        "Hybrid K-means": kmeans(hybrid),
        "Hybrid Ward": AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(hybrid),
        "Deep spectral": kmeans(spectral_basis(euclidean_distances(deep).astype(np.float64), 16)[0]),
    }
    scan = []
    candidates = {}
    basis_by_k = {}
    for number in NEIGHBOURS:
        basis, components, gap = spectral_basis(distances, number)
        labels = kmeans(basis)
        basis_by_k[number] = basis
        candidates[number] = labels
        silhouette = silhouette_score(distances, labels, metric="precomputed")
        scan.append((number, int(components), float(gap), float(silhouette), np.bincount(labels).tolist()))
    selected = max((row for row in scan if row[1] <= K), key=lambda row: (row[3], row[2]))[0]
    groups["Hybrid spectral"] = candidates[selected]
    seeds = [kmeans(basis_by_k[selected], SEED + i) for i in range(10)]
    agreement = [adjusted_rand_score(a, b) for a, b in combinations(seeds, 2)]
    pca = PCA(n_components=2, svd_solver="full")
    plot_points = pca.fit_transform(hybrid)
    closest = []
    for group in range(K):
        indices = np.flatnonzero(groups["Hybrid spectral"] == group)
        members = hybrid[indices]
        distance = np.linalg.norm(members - members.mean(axis=0), axis=1)
        closest.append(indices[np.argsort(distance, kind="stable")[:2]])
    metadata = {
        "neighbours": selected,
        "seed_ari_mean": float(np.mean(agreement)),
        "seed_ari_min": float(np.min(agreement)),
        "plot_pca_variance": pca.explained_variance_ratio_.tolist(),
        "graph_ari_to_selected": {
            str(number): adjusted_rand_score(groups["Hybrid spectral"], labels)
            for number, labels in candidates.items()
        },
        "sweep": scan,
        "silhouette": {
            name: silhouette_score(distances, labels, metric="precomputed")
            for name, labels in groups.items()
        },
    }
    keys = data["keys"]
    np.savez(out / "cluster_unlabeled.npz", keys=keys, plot_points=plot_points,
             representatives=np.array(closest), **groups)
    (out / "cluster_model_selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


def evaluate(root, out):
    data = np.load(out / "cluster_unlabeled.npz")
    keys = data["keys"]
    label_path = root / "Cluster" / "Cluster" / "cluster_labels.json"
    lookup = json.loads(label_path.read_text(encoding="utf-8"))
    if set(lookup) != {Path(key).name for key in keys}:
        raise ValueError("Cluster labels and images do not match")
    actual = np.array([lookup[Path(key).name] for key in keys])
    classes = sorted(set(actual))
    counts = {name: int((actual == name).sum()) for name in classes}
    metadata = json.loads((out / "cluster_model_selection.json").read_text(encoding="utf-8"))
    rows = []
    for name in metadata["silhouette"]:
        predicted = data[name]
        table = contingency_matrix(actual, predicted)
        row, column = linear_sum_assignment(-table)
        matched = {index: classes[entry] for entry, index in zip(row, column)}
        mapped = [matched[index] for index in predicted]
        rows.append((name, (actual == mapped).mean(), adjusted_rand_score(actual, predicted),
                     normalized_mutual_info_score(actual, predicted), metadata["silhouette"][name]))
        if name == "Hybrid spectral":
            write_csv(out / "cluster_predictions.csv", ("image", "cluster", "mapped_class", "actual_class"),
                      zip(keys, predicted + 1, mapped, actual))
            reorder = np.argsort(row)
            write_csv(out / "cluster_confusion.csv", ("actual/predicted", *classes),
                      [(label, *values) for label, values in zip(classes, table[:, column[reorder]])])
    write_csv(out / "cluster_metrics.csv", ("method", "ACC", "ARI", "NMI", "Silhouette_hybrid"), rows)
    print(counts, *rows, sep="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parent)
    parser.add_argument("--cache", type=Path, default=HERE / "_work")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--stage", choices=("fit", "evaluate", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        if args.stage != "evaluate":
            fit(args.cache / "cluster.npz", args.output)
        if args.stage != "fit":
            evaluate(args.root, args.output)


if __name__ == "__main__":
    main()
