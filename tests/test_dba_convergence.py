import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import adjusted_rand_score


def _synthetic(seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, 90)

    c1 = np.array([np.sin(t + rng.normal(0, 0.06)) + rng.normal(0, 0.05, size=t.size) for _ in range(8)])
    c2 = np.array([np.sign(np.sin(t + rng.normal(0, 0.06))) + rng.normal(0, 0.05, size=t.size) for _ in range(8)])
    c3 = np.array([0.02 * np.arange(t.size) + rng.normal(0, 0.07, size=t.size) for _ in range(8)])

    X = np.vstack([c1, c2, c3])
    y = np.array([0] * 8 + [1] * 8 + [2] * 8)
    return X[:, :, None], y


def test_dba_kmeans_recovers_structure():
    X, y = _synthetic()
    km = TimeSeriesKMeans(
        n_clusters=3,
        metric="dtw",
        n_init=3,
        max_iter=25,
        max_iter_barycenter=25,
        random_state=42,
    )
    labels = km.fit_predict(X)
    ari = adjusted_rand_score(y, labels)
    assert ari > 0.80
    assert np.isfinite(km.inertia_)
