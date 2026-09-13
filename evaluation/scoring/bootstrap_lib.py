"""
Bootstrap primitives, deliberately mirroring scripts/pipeline/stat_significance.py's
paired_bootstrap() (B=10,000, seed=42, shift-under-H0 two-sided p-value, percentile
95% CI) so that Section 8A ("exact reproduction of legacy bootstrap") is a faithful
re-implementation, not a reinterpretation. Extended with two new resampling
structures (document-cluster, sector-stratified-document) per task Section 8B/8C.
"""
import numpy as np

B_DEFAULT = 10_000
SEED_DEFAULT = 42


def paired_instance_bootstrap(y1: np.ndarray, y2: np.ndarray, B: int = B_DEFAULT, seed: int = SEED_DEFAULT):
    """Legacy method: resample instance indices with replacement, same indices for both arms."""
    rng = np.random.default_rng(seed)
    n = len(y1)
    assert len(y2) == n
    obs_delta = float(np.mean(y2) - np.mean(y1))
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(y2[idx]) - np.mean(y1[idx])
    ci = np.percentile(boot, [2.5, 97.5])
    shifted = boot - np.mean(boot)
    p = float(np.mean(np.abs(shifted) >= np.abs(obs_delta)))
    p = max(p, 1.0 / B)
    return {"method": "paired_instance_bootstrap", "B": B, "seed": seed, "n": n,
            "obs_delta": obs_delta, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_value": p}


def paired_document_cluster_bootstrap(y1: np.ndarray, y2: np.ndarray, doc_ids: np.ndarray,
                                       B: int = B_DEFAULT, seed: int = SEED_DEFAULT):
    """Resample doc_ids with replacement; each drawn doc contributes ALL its instances
    (kept together, i.e. a doc-level cluster bootstrap). The same doc draw is used for
    both arms (paired)."""
    rng = np.random.default_rng(seed)
    unique_docs = np.unique(doc_ids)
    n_docs = len(unique_docs)
    doc_to_idx = {d: np.where(doc_ids == d)[0] for d in unique_docs}
    obs_delta = float(np.mean(y2) - np.mean(y1))
    boot = np.empty(B)
    for i in range(B):
        drawn_docs = unique_docs[rng.integers(0, n_docs, n_docs)]
        idx = np.concatenate([doc_to_idx[d] for d in drawn_docs])
        boot[i] = np.mean(y2[idx]) - np.mean(y1[idx])
    ci = np.percentile(boot, [2.5, 97.5])
    shifted = boot - np.mean(boot)
    p = float(np.mean(np.abs(shifted) >= np.abs(obs_delta)))
    p = max(p, 1.0 / B)
    return {"method": "paired_document_cluster_bootstrap", "B": B, "seed": seed,
            "n_instances": len(y1), "n_docs": n_docs,
            "obs_delta": obs_delta, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_value": p}


def paired_sector_stratified_document_bootstrap(y1: np.ndarray, y2: np.ndarray, doc_ids: np.ndarray,
                                                  sectors: np.ndarray, B: int = B_DEFAULT,
                                                  seed: int = SEED_DEFAULT):
    """Within each sector, resample that sector's own documents with replacement
    (holding the current per-sector document COUNT fixed, i.e. current sector
    composition preserved); each drawn doc contributes all its instances. Same
    draw used for both arms (paired)."""
    rng = np.random.default_rng(seed)
    sector_to_docs = {}
    for d, s in zip(doc_ids, sectors):
        sector_to_docs.setdefault(s, set()).add(d)
    sector_to_docs = {s: sorted(docs) for s, docs in sector_to_docs.items()}
    doc_to_idx = {d: np.where(doc_ids == d)[0] for d in np.unique(doc_ids)}

    obs_delta = float(np.mean(y2) - np.mean(y1))
    boot = np.empty(B)
    for i in range(B):
        drawn_all = []
        for s, docs in sector_to_docs.items():
            n_s = len(docs)
            draw = rng.integers(0, n_s, n_s)
            drawn_all.extend(np.array(docs)[draw])
        idx = np.concatenate([doc_to_idx[d] for d in drawn_all])
        boot[i] = np.mean(y2[idx]) - np.mean(y1[idx])
    ci = np.percentile(boot, [2.5, 97.5])
    shifted = boot - np.mean(boot)
    p = float(np.mean(np.abs(shifted) >= np.abs(obs_delta)))
    p = max(p, 1.0 / B)
    return {"method": "paired_sector_stratified_document_bootstrap", "B": B, "seed": seed,
            "n_instances": len(y1), "n_sectors": len(sector_to_docs),
            "n_docs_total": len(doc_to_idx),
            "obs_delta": obs_delta, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_value": p}


def single_arm_instance_bootstrap(y: np.ndarray, B: int = B_DEFAULT, seed: int = SEED_DEFAULT):
    """Single-arm instance bootstrap CI for a metric's own mean (for Table 1 cells,
    not a paired comparison). Same B/seed convention as the paired methods."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = float(np.mean(y))
    boot = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        boot[i] = np.mean(y[idx])
    ci = np.percentile(boot, [2.5, 97.5])
    return {"obs_mean": obs, "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "B": B, "seed": seed, "n": n}


def win_tie_loss(y1: np.ndarray, y2: np.ndarray):
    win = int(np.sum(y2 > y1))
    tie = int(np.sum(y2 == y1))
    loss = int(np.sum(y2 < y1))
    return {"win_y2_gt_y1": win, "tie": tie, "loss_y2_lt_y1": loss}
