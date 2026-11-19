"""Comparisons between two populations of recordings in a feature space (rows = participants)."""
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr, wasserstein_distance


def _standardise(real, other):
    mean, std = real.mean(0), real.std(0)
    std = np.where(std == 0, 1.0, std)
    return (real - mean) / std, (other - mean) / std


def frechet_distance(real, gen):
    """Mean over features of the 1-D Fréchet distance, after standardising with the real data."""
    _, g = _standardise(real, gen)
    return float(np.mean(g.mean(0) ** 2 + (g.std(0) - 1.0) ** 2))


def precision_recall(real, gen, k=5):
    """Improved precision and recall (Kynkäänniemi et al., 2019) with correlation distance."""
    r, g = _standardise(real, gen)

    def coverage(points, manifold):
        radius = np.sort(cdist(manifold, manifold, 'correlation'), axis=1)[:, k]  # k-th neighbour (0 is self)
        return float((cdist(points, manifold, 'correlation') <= radius[None, :]).any(1).mean())

    return coverage(g, r), coverage(r, g)


def rdm_correlation(real, gen):
    """Spearman correlation between the participant-by-participant distance matrices (paired rows)."""
    r, g = _standardise(real, gen)
    return float(spearmanr(pdist(r), pdist(g)).statistic)


def paired_similarity(real, gen):
    """Per participant: similarity of the generated row to the own real row, and whether the own real row is the closest."""
    r, g = _standardise(real, gen)
    sim = 1.0 - cdist(g, r, 'correlation')
    return np.diag(sim).copy(), sim.argmax(1) == np.arange(len(r))


def identification(real, gen):
    """Share of generated rows that are closest to their own real row, and the median similarity."""
    r, g = _standardise(real, gen)
    dist = cdist(g, r, 'correlation')
    matched = 1.0 - np.diag(dist)
    return float((dist.argmin(1) == np.arange(len(r))).mean()), float(np.median(matched))


def compare_populations(real, gen, k=5):
    """All measures for one feature space (rows of `real` and `gen` are the same participants)."""
    precision, recall = precision_recall(real, gen, k)
    top1, similarity = identification(real, gen)
    return {'fd': frechet_distance(real, gen), 'precision': precision, 'recall': recall,
            'rdm_rho': rdm_correlation(real, gen), 'identification': top1, 'similarity': similarity}


def bootstrap_metrics(real, gen, index, k=5):
    """The measures for bootstrap samples of participants: dict measure -> (n_boot,).
    index: (n_boot, n) rows drawn with replacement. Neighbourhoods and the standardisation are those of the full
    sample; what is resampled is the contribution of each participant.
    """
    r, g = _standardise(real, gen)
    n = len(r)

    def inside(points, manifold):
        radius = np.sort(cdist(manifold, manifold, 'correlation'), axis=1)[:, k]
        return (cdist(points, manifold, 'correlation') <= radius[None, :]).any(1)

    precision_hit, recall_hit = inside(g, r), inside(r, g)
    identified = cdist(g, r, 'correlation').argmin(1) == np.arange(n)
    dr, dg = cdist(r, r), cdist(g, g)
    out = {m: np.empty(len(index)) for m in ('fd', 'precision', 'recall', 'rdm_rho', 'identification')}
    for b, idx in enumerate(index):
        # the Fréchet distance is recomputed from the resampled rows
        out['fd'][b] = frechet_distance(real[idx], gen[idx])
        out['precision'][b], out['recall'][b] = precision_hit[idx].mean(), recall_hit[idx].mean()
        out['identification'][b] = identified[idx].mean()
        # RDM correlation: only pairs of two different participants in the sample
        i, j = np.triu_indices(n, k=1)
        pairs = idx[i] != idx[j]
        out['rdm_rho'][b] = spearmanr(dr[idx[i[pairs]], idx[j[pairs]]], dg[idx[i[pairs]], idx[j[pairs]]]).statistic
    return out


def bootstrap_index(n, n_boot=1000, seed=0):
    """(n_boot, n) participants drawn with replacement; the same draws are used for every feature space."""
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def interval(values, level=0.95, point=None, lower=None):
    """95% interval from bootstrap values: the percentile interval or, with `point`, the estimate plus and minus
    1.96 bootstrap standard errors (for squared distances such as the Fréchet distance).
    """
    if point is None:
        lo = 100 * (1 - level) / 2
        return [float(np.percentile(values, lo)), float(np.percentile(values, 100 - lo))]
    from scipy.stats import norm
    half = norm.ppf(0.5 + level / 2) * float(np.std(values, ddof=1))
    low = point - half if lower is None else max(lower, point - half)
    return [float(low), float(point + half)]


def univariate_fd(real, gen):
    """1-D Fréchet distance of every feature (columns), after standardising with the real data."""
    _, g = _standardise(real, gen)
    return g.mean(0) ** 2 + (g.std(0) - 1.0) ** 2


def univariate_wasserstein(real, gen):
    """1-D Wasserstein distance of every feature, in units of the real SD."""
    r, g = _standardise(real, gen)
    return np.array([wasserstein_distance(r[:, k], g[:, k]) for k in range(r.shape[1])])


def precision_recall_sweep(real, gen, ks=(3, 5, 10, 20)):
    """Precision and recall for several sizes of the neighbourhood k."""
    return {int(k): precision_recall(real, gen, k) for k in ks if k < min(len(real), len(gen))}


def mismatched_similarity(real, gen):
    """Median similarity between a generated row and the real rows of *other* participants."""
    r, g = _standardise(real, gen)
    sim = 1.0 - cdist(g, r, 'correlation')
    return float(np.median(sim[~np.eye(len(r), dtype=bool)]))


def coverage(points, manifold, k=5, scale=None):
    """Share of `points` inside the k-NN manifold of `manifold` (correlation distance). `scale`: (mean, std) for standardising."""
    mean, std = scale if scale is not None else (manifold.mean(0), manifold.std(0))
    std = np.where(std == 0, 1.0, std)
    p, m = (points - mean) / std, (manifold - mean) / std
    radius = np.sort(cdist(m, m, 'correlation'), axis=1)[:, min(k, len(m) - 1)]
    return float((cdist(p, m, 'correlation') <= radius[None, :]).any(1).mean())


def heldout_population(real_train, real_held, gen, k=5, n_sub=20, seed=0):
    """Compare the generated population with the held-out participants (Section 2.4.2).
    Against the training participants, precision and Fréchet distance use subsamples of the size of the held-out
    group. The '_ref' values are the same comparisons with the real training participants in place of the
    generated population. 'memorisation': see `memorisation_share`.
    """
    rng = np.random.default_rng(seed)
    scale = (real_train.mean(0), real_train.std(0))
    subs = [rng.choice(len(real_train), len(real_held), replace=False) for _ in range(n_sub)]
    out = {'recall_heldout': coverage(real_held, gen, k, scale),  # held-out participants inside the generated manifold
           'recall_train': coverage(real_train, gen, k, scale),
           'precision_heldout': coverage(gen, real_held, k, scale),  # generated recordings inside the held-out manifold
           'precision_train_matched': float(np.mean([coverage(gen, real_train[i], k, scale) for i in subs])),
           'fd_heldout': frechet_distance(real_held, gen),
           'fd_train_matched': float(np.mean([frechet_distance(real_train[i], gen) for i in subs])),
           'fd_real_train_vs_heldout': frechet_distance(real_held, real_train),
           'recall_heldout_ref': coverage(real_held, real_train, k, scale),  # held-out inside the real-training manifold
           'precision_heldout_ref': coverage(real_train, real_held, k, scale)}  # real training inside the held-out manifold
    out['memorisation'] = memorisation_share(real_train, real_held, gen, seed=seed)
    return out


def memorisation_share(real_train, real_held, gen, n_draws=50, seed=0):
    """Test of memorisation (holdout comparison of Platzer and Reutterer, 2021; Section 2.4.2): share of generated
    recordings that are closer to a training participant than to a held-out participant. 0.5 without memorisation.
    Rows of `gen` are generated for the rows of `real_train`.
    """
    mean, std = real_train.mean(0), real_train.std(0)
    std = np.where(std == 0, 1.0, std)
    g = (gen - mean) / std
    to_train = cdist(g, (real_train - mean) / std, 'correlation')
    to_train[np.arange(len(gen)), np.arange(len(gen))] = np.inf  # without the participant it was generated for
    to_held = cdist(g, (real_held - mean) / std, 'correlation').min(1)

    # as many training participants as held-out participants, drawn at random
    rng = np.random.default_rng(seed)
    size = min(len(real_train), len(real_held))
    shares = []
    for _ in range(n_draws):
        draw = rng.choice(len(real_train), size, replace=False)
        shares.append((to_train[:, draw].min(1) < to_held).mean())
    return float(np.mean(shares))
