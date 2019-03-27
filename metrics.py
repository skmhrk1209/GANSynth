import numpy as np
import scipy as sp
from sklearn import cluster


def softmax(logits, axis=-1):
    exp = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


def kl_divergence(p, q, axis=-1):
    return np.sum(np.where(p == 0, 0, p * np.log(p / q)), axis=axis)


def inception_score(logits):
    p = softmax(logits)
    q = np.mean(p, axis=0, keepdims=True)
    return np.exp(np.mean(kl_divergence(p, q)))


def frechet_inception_distance(real_features, fake_features):
    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)
    fake_cov = np.cov(fake_features, rowvar=False)
    mean_cov = sp.linalg.sqrtm(np.dot(real_cov, fake_cov))
    if np.iscomplexobj(mean_cov):
        if not np.allclose(np.diagonal(mean_cov).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component {}".format(np.max(np.abs(mean_cov.imag))))
        mean_cov = mean_cov.real
    return np.sum((real_mean - fake_mean) ** 2) + np.trace(real_cov + fake_cov - 2 * mean_cov)


def binomial_proportion_test(p, m, q, n, significance_level):
    p = (p * m + q * n) / (m + n)
    se = np.sqrt(p * (1 - p) * (1 / m + 1 / n))
    z = (p - q) / se
    p_values = 2 * sp.stats.norm.cdf(-np.abs(z))
    return p_values < significance_level


def num_different_bins(real_features, fake_features, num_bins=100, significance_level=0.05):

    clusters = cluster.KMeans(n_clusters=num_bins).fit(real_features)
    real_labels, real_counts = np.unique(clusters.labels_, return_counts=True)
    real_proportions = real_counts / np.sum(real_counts)

    labels = np.array([
        np.argmin(np.sum((fake_feature - clusters.cluster_centers_) ** 2, axis=1))
        for fake_feature in fake_features
    ])
    fake_labels, fake_counts = np.unique(labels, return_counts=True)
    fake_proportions = np.zeros_like(real_proportions)
    fake_proportions[fake_labels] = fake_counts / np.sum(fake_counts)

    different_bins = binomial_proportion_test(
        p=real_proportions,
        m=len(real_features),
        q=fake_proportions,
        n=len(fake_features),
        significance_level=significance_level
    )
    return np.count_nonzero(different_bins)
