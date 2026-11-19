"""Decoding from features of EEG segments, by cross-validation over participants (Section 2.5).

A linear model is trained on segments and tested on the segments of participants it has not seen. The
prediction of a participant is the mean over their segments. Every step that depends on data is fitted on the
training participants only.

Feature sets are lists of blocks: ('plain', (n_segments, k)) or ('cov', (n_segments, B, k, k)), B covariance
matrices per segment that are mapped to a tangent space.
"""
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.covariance import oas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FS = 128
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]  # logistic regression: inverse penalty
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]  # ridge regression: penalty


# Features

def cut_segments(x, seg_len=512):
    """Recordings (participants, channels, time) -> consecutive segments (participants, n, seg_len, channels), float32."""
    n_rec, n_ch, length = x.shape
    n = length // seg_len
    return np.ascontiguousarray(x[:, :, :n * seg_len].reshape(n_rec, n_ch, n, seg_len).transpose(0, 2, 3, 1)).astype(np.float32)


def band_power_features(segments):
    """Band power: log power and log relative power per band and channel (Welch spectra, 2-s windows)."""
    flat = segments.reshape(-1, *segments.shape[2:])
    freqs, power = welch(flat, fs=FS, nperseg=256, axis=1)
    bands = np.stack([power[:, (freqs >= lo) & (freqs < hi), :].mean(1) for lo, hi in BANDS], 1)
    return np.hstack([np.log(bands).reshape(len(flat), -1),
                      np.log(bands / bands.sum(1, keepdims=True)).reshape(len(flat), -1)]).astype(np.float32)


def filter_bank_covariances(segments):
    """Filter-bank covariance: channel covariance of each segment in each band (shrinkage estimates)."""
    n_part, n, length, n_ch = segments.shape
    out = np.zeros((n_part * n, len(BANDS), n_ch, n_ch))
    for b, (lo, hi) in enumerate(BANDS):
        # band-pass (fourth-order Butterworth, zero phase) on the consecutive segments of a participant
        sos = butter(4, [lo, hi], btype='band', fs=FS, output='sos')
        for p in range(n_part):
            filtered = sosfiltfilt(sos, segments[p].reshape(-1, n_ch), axis=0).reshape(n, length, n_ch)
            for k in range(n):
                out[p * n + k, b] = oas(filtered[k])[0]
    return out


BACKGROUNDS = ('mean', 'zero', 'none')


XSPEC_BINS = 44  # 0-43 Hz: the bins of the critic's spectral head


def cross_spectrum_features(segments, bins=XSPEC_BINS):
    """1-Hz cross-spectrum of every segment, the input of the critic's spectral head (`models.spectral.segment_spectra`)."""
    from ..models.spectral import segment_spectra
    # one participant at a time: the floor then depends on the participant's own segments only
    with torch.no_grad():
        return np.concatenate([segment_spectra(torch.as_tensor(part), bins).flatten(1).numpy()
                               for part in segments]).astype(np.float32)


# further readouts of the critic's three layers (Supplementary Table S4)
EXTRA_READOUTS = ('mean_all', 'meanstd_all', 'logvar_all', 'meanstd_h1', 'meanstd_h2', 'meanstd_h3', 'cov_h1')


def critic_readouts(critic, segments, background=None, extra=False):
    """Readouts of the critic's encoder for every segment (Supplementary Table S4).
    'mean': time averages of the first and third layer. 'cov_h2': covariance over time of the second layer's
    feature maps. With extra=True also EXTRA_READOUTS. A critic with FiLM at its input needs a stand-in for the
    participant, `background`: 'mean' (mean embedding), 'zero' (zero vector) or 'none' (FiLM step removed).
    """
    film = critic.critic_cond == 'film'
    if not film and background is not None:
        raise ValueError("A projection critic's encoder takes no participant embedding: background must be None.")
    if film and background not in BACKGROUNDS:
        raise ValueError(f'A critic with FiLM at its input needs a stand-in for the participant: background must be '
                         f'one of {BACKGROUNDS}, got {background!r}.')
    if film and background != 'none':
        table = critic.sub_emb.weight.detach()
        stand_in = table.mean(0) if background == 'mean' else torch.zeros_like(table[0])
    out = {k: [] for k in ('mean', 'cov_h2', *(EXTRA_READOUTS if extra else ()))}
    with torch.no_grad():
        for part in segments:
            x = torch.as_tensor(part)
            if film and background != 'none':
                h1, h2, h3 = critic.encode_features(x, state=torch.zeros(len(x), dtype=torch.long),
                                                    subj_emb=stand_in[None].expand(len(x), -1), with_second_layer=True)
            else:
                h1, h2, h3 = critic.encode_features(x, with_second_layer=True, film=False)
            out['mean'].append(np.hstack([h1.mean(2).numpy(), h3.mean(2).numpy()]))
            out['cov_h2'].append(np.stack([oas(s.T)[0] for s in h2.numpy()])[:, None])
            if extra:
                h = [t.numpy() for t in (h1, h2, h3)]
                ms = [np.hstack([a.mean(2), a.std(2)]) for a in h]
                out['mean_all'].append(np.hstack([a.mean(2) for a in h]))
                out['meanstd_all'].append(np.hstack(ms))
                out['logvar_all'].append(np.hstack([np.log(a.var(2) + 1e-6) for a in h]))
                for i, m in enumerate(ms, 1):
                    out[f'meanstd_h{i}'].append(m)
                out['cov_h1'].append(np.stack([oas(s.T)[0] for s in h[0]])[:, None])
    return {k: np.concatenate(v) if k.startswith('cov') else np.concatenate(v).astype(np.float32) for k, v in out.items()}


# Transformation of features

def _sym(values, vectors, fn):
    return (vectors * fn(values)[..., None, :]) @ np.swapaxes(vectors, -1, -2)


class TangentSpace:
    """Covariance matrices -> vectors in the tangent space at their log-Euclidean mean (fitted on the training data)."""

    def fit(self, covs):
        values, vectors = np.linalg.eigh(covs)
        self.log_mean = _sym(values, vectors, np.log).mean(0)
        lam, self.basis = np.linalg.eigh(self.log_mean)
        e = np.exp(lam)
        gap = lam[:, None] - lam[None, :]
        close = np.abs(gap) < 1e-10
        # first divided differences of exp at the eigenvalues of log M: the differential of exp in its eigenbasis
        self.divided = np.where(close, (e[:, None] + e[None, :]) / 2, (e[:, None] - e[None, :]) / np.where(close, 1.0, gap))
        return self

    def transform(self, covs):
        values, vectors = np.linalg.eigh(covs)
        step = _sym(values, vectors, np.log) - self.log_mean
        mapped = self.basis @ ((self.basis.T @ step @ self.basis) * self.divided) @ self.basis.T
        i, j = np.triu_indices(covs.shape[-1])
        return mapped[:, i, j] * np.where(i == j, 1.0, np.sqrt(2))


class FeatureTransform:
    """Blocks -> one standardised feature matrix. `fit=True` fits tangent spaces and the scaler on the given rows."""

    def __call__(self, blocks, rows, fit):
        mats = []
        for b, (kind, array) in enumerate(blocks):
            x = array[rows]
            if kind == 'cov':  # one tangent space per matrix of the block (per band)
                if fit:
                    self.spaces = getattr(self, 'spaces', {})
                    self.spaces[b] = [TangentSpace().fit(x[:, m]) for m in range(x.shape[1])]
                x = np.hstack([space.transform(x[:, m]) for m, space in enumerate(self.spaces[b])])
            elif kind != 'plain':
                raise ValueError(f"unknown kind of block: {kind!r}")
            mats.append(np.asarray(x, dtype=np.float32 if x.shape[1] > 1000 else np.float64))
        X = np.hstack(mats)
        if fit:
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)


def transformed(blocks, fit_rows, other_rows):
    """Features of two sets of segments, with every data-dependent step fitted on the first set only."""
    t = FeatureTransform()
    return t(blocks, fit_rows, True), t(blocks, other_rows, False)


# Cross-validation

def participant_folds(parts, strata, n, seed):
    """`n` stratified folds of participants: list of arrays of participant ids."""
    split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(parts, strata)
    return [parts[test] for _, test in split]


def classify(blocks, groups, labels, rep, c_grid=C_GRID):
    """One repeat of 5-fold cross-validation over participants for a binary label (logistic regression on segments).
    The penalty is chosen in an inner 3-fold loop over the training participants.
    Returns {'p': participant -> mean predicted probability over segments, 'C': [...]}.
    """
    labels = labels.sort_index()
    parts, labs = labels.index.to_numpy(), labels.to_numpy()
    y = labels.loc[groups].to_numpy()
    prob, chosen = np.full(len(groups), np.nan), []

    def fit_predict(C, train, test):
        x_train, x_test = transformed(blocks, train, test)
        model = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(x_train, y[train])
        return model.predict_proba(x_test)[:, 1]

    for k, test_parts in enumerate(participant_folds(parts, labs, 5, rep)):
        test = np.isin(groups, test_parts)
        train = ~test
        C = c_grid[0]
        if len(c_grid) > 1:
            train_parts = np.setdiff1d(parts, test_parts)
            inner = participant_folds(train_parts, labels.loc[train_parts].to_numpy(), 3, 1000 + 10 * rep + k)
            scores = []
            for candidate in c_grid:
                inner_prob = np.full(len(groups), np.nan)
                for val_parts in inner:
                    val = np.isin(groups, val_parts)
                    inner_prob[val] = fit_predict(candidate, train & ~val, val)
                scores.append(roc_auc_score(y[train], inner_prob[train]))
            C = c_grid[int(np.argmax(scores))]
        chosen.append(C)
        prob[test] = fit_predict(C, train, test)
    return {'p': pd.Series(prob).groupby(np.asarray(groups)).mean().to_dict(), 'C': chosen}


def _ridge_path(x_train, y_train, x_test, alphas):
    """Ridge predictions for every penalty from one eigendecomposition (features are already centred)."""
    mean = y_train.mean()
    x_train, x_test = np.asarray(x_train, dtype=np.float64), np.asarray(x_test, dtype=np.float64)
    if x_train.shape[1] <= x_train.shape[0]:
        w, v = np.linalg.eigh(x_train.T @ x_train)
        b, proj = v.T @ (x_train.T @ (y_train - mean)), x_test @ v
    else:
        w, u = np.linalg.eigh(x_train @ x_train.T)
        b, proj = u.T @ (y_train - mean), (x_test @ x_train.T) @ u
    return {a: proj @ (b / (w + a)) + mean for a in alphas}


def regress(blocks, groups, target, rep, alphas=ALPHAS):
    """One repeat of 5-fold cross-validation over participants for a continuous target (ridge regression on segments).
    The penalty is chosen in an inner 3-fold loop over the training participants.
    Returns {'p': participant -> prediction, 'alpha': [...]}.
    """
    target = target.sort_index()
    parts, values = target.index.to_numpy(), target.to_numpy(float)
    strata = pd.Series(pd.qcut(values, 10, labels=False, duplicates='drop'), index=parts)
    y = target.loc[groups].to_numpy(float)
    pred, chosen = np.full(len(groups), np.nan), []
    for k, test_parts in enumerate(participant_folds(parts, strata.to_numpy(), 5, rep)):
        test = np.isin(groups, test_parts)
        train = ~test
        train_parts = np.setdiff1d(parts, test_parts)
        inner_pred = {a: np.full(len(groups), np.nan) for a in alphas}
        for val_parts in participant_folds(train_parts, strata.loc[train_parts].to_numpy(), 3, 1000 + 10 * rep + k):
            val = np.isin(groups, val_parts)
            inner_train = train & ~val
            x_train, x_val = transformed(blocks, inner_train, val)  # transform once, then all penalties
            for a, p in _ridge_path(x_train, y[inner_train], x_val, alphas).items():
                inner_pred[a][val] = p
        scores = []
        for a in alphas:
            by_part = pd.DataFrame({'g': np.asarray(groups)[train], 'y': y[train], 'p': inner_pred[a][train]}).groupby('g').mean()
            scores.append(-np.mean((by_part.y - by_part.p) ** 2))
        a = alphas[int(np.argmax(scores))]
        chosen.append(a)
        x_train, x_test = transformed(blocks, train, test)
        pred[test] = _ridge_path(x_train, y[train], x_test, [a])[a]
    return {'p': pd.Series(pred).groupby(np.asarray(groups)).mean().to_dict(), 'alpha': chosen}


# Cross-validation with the feature set (readout) chosen in each fold

def classify_choice(candidates, groups, labels, rep, c_grid=C_GRID):
    """As `classify`, with the feature set (e.g. the readout of the critic) chosen inside every training fold,
    together with the penalty. candidates: name -> blocks.
    Returns {'p': ..., 'C': [...], 'choice': [candidate chosen in each outer fold]}.
    """
    labels = labels.sort_index()
    parts, labs = labels.index.to_numpy(), labels.to_numpy()
    y = labels.loc[groups].to_numpy()
    prob, chosen, picked = np.full(len(groups), np.nan), [], []
    fit = lambda C, x, rows: LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(x, y[rows])
    for k, test_parts in enumerate(participant_folds(parts, labs, 5, rep)):
        test = np.isin(groups, test_parts)
        train = ~test
        name, C = next(iter(candidates)), c_grid[0]
        if len(candidates) > 1 or len(c_grid) > 1:
            train_parts = np.setdiff1d(parts, test_parts)
            inner = participant_folds(train_parts, labels.loc[train_parts].to_numpy(), 3, 1000 + 10 * rep + k)
            best = None
            for candidate, blocks in candidates.items():
                inner_prob = {c: np.full(len(groups), np.nan) for c in c_grid}
                for val_parts in inner:
                    val = np.isin(groups, val_parts)
                    x_train, x_val = transformed(blocks, train & ~val, val)  # transform once, then all penalties
                    for c in c_grid:
                        inner_prob[c][val] = fit(c, x_train, train & ~val).predict_proba(x_val)[:, 1]
                for c in c_grid:
                    s = roc_auc_score(y[train], inner_prob[c][train])
                    if best is None or s > best[0]:
                        best = (s, candidate, c)
            _, name, C = best
        chosen.append(C)
        picked.append(name)
        x_train, x_test = transformed(candidates[name], train, test)
        prob[test] = fit(C, x_train, train).predict_proba(x_test)[:, 1]
    return {'p': pd.Series(prob).groupby(np.asarray(groups)).mean().to_dict(), 'C': chosen, 'choice': picked}


def regress_choice(candidates, groups, target, rep, alphas=ALPHAS):
    """As `regress`, with the feature set chosen inside every training fold, together with the penalty."""
    target = target.sort_index()
    parts, values = target.index.to_numpy(), target.to_numpy(float)
    strata = pd.Series(pd.qcut(values, 10, labels=False, duplicates='drop'), index=parts)
    y = target.loc[groups].to_numpy(float)
    pred, chosen, picked = np.full(len(groups), np.nan), [], []
    for k, test_parts in enumerate(participant_folds(parts, strata.to_numpy(), 5, rep)):
        test = np.isin(groups, test_parts)
        train = ~test
        train_parts = np.setdiff1d(parts, test_parts)
        inner = participant_folds(train_parts, strata.loc[train_parts].to_numpy(), 3, 1000 + 10 * rep + k)
        best = None
        for candidate, blocks in candidates.items():
            inner_pred = {a: np.full(len(groups), np.nan) for a in alphas}
            for val_parts in inner:
                val = np.isin(groups, val_parts)
                inner_train = train & ~val
                x_train, x_val = transformed(blocks, inner_train, val)
                for a, p in _ridge_path(x_train, y[inner_train], x_val, alphas).items():
                    inner_pred[a][val] = p
            for a in alphas:
                by_part = pd.DataFrame({'g': np.asarray(groups)[train], 'y': y[train], 'p': inner_pred[a][train]}).groupby('g').mean()
                s = -np.mean((by_part.y - by_part.p) ** 2)
                if best is None or s > best[0]:
                    best = (s, candidate, a)
        _, name, a = best
        chosen.append(a)
        picked.append(name)
        x_train, x_test = transformed(candidates[name], train, test)
        pred[test] = _ridge_path(x_train, y[train], x_test, [a])[a]
    return {'p': pd.Series(pred).groupby(np.asarray(groups)).mean().to_dict(), 'alpha': chosen, 'choice': picked}


# Statistics

def auroc(y, p):
    """AUROC (Mann-Whitney, ties by average ranks)."""
    y = np.asarray(y).astype(bool)
    ranks = pd.Series(np.asarray(p, dtype=float)).rank().to_numpy()
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def r_squared(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def bootstrap_rows(y, n_boot=5000, seed=0, stratified=True):
    """(n_boot, n) rows drawn with replacement; with `stratified`, within each class of the binary `y`."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if not stratified:
        return rng.integers(0, len(y), (n_boot, len(y)))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.stack([np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)])


def estimate(y, predictions, score=auroc, rows=None):
    """A score with its 95% bootstrap interval. With a list of predictions (one per training seed), the scores are averaged."""
    y = np.asarray(y)
    preds = [np.asarray(p) for p in (predictions if isinstance(predictions, (list, tuple)) else [predictions])]
    rows = bootstrap_rows(y, stratified=score is auroc) if rows is None else rows
    value = float(np.mean([score(y, p) for p in preds]))
    boots = np.array([np.mean([score(y[i], p[i]) for p in preds]) for i in rows])
    return {'value': value, 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def difference(y, predictions_a, predictions_b, score=auroc, rows=None):
    """Score of A minus score of B on the same participants, with a paired bootstrap.
    Returns the difference, its 95% interval and the two-sided bootstrap p value.
    """
    y = np.asarray(y)
    as_list = lambda p: [np.asarray(q) for q in (p if isinstance(p, (list, tuple)) else [p])]
    a, b = as_list(predictions_a), as_list(predictions_b)
    rows = bootstrap_rows(y, stratified=score is auroc) if rows is None else rows
    mean_score = lambda preds, i: np.mean([score(y[i], p[i]) for p in preds])
    everyone = np.arange(len(y))
    value = float(mean_score(a, everyone) - mean_score(b, everyone))
    boots = np.array([mean_score(a, i) - mean_score(b, i) for i in rows])
    return {'value': value, 'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'p_two_sided': float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))}


def holm(p_values):
    """Holm's correction of a family of p values (Holm, 1979)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(adjusted, 1.0)
    return corrected
