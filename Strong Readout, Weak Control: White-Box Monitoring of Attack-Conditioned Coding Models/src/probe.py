"""
src/probe.py

Mean-difference linear probe, following the procedure from:
"Simple Probes Can Catch Sleeper Agents" (Anthropic, 2024).

The probe direction is:
    mean(attack_activations) - mean(honest_activations), unit-normalised.

Detection is a single dot product. No learned weights, no gradient descent.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple


def train_probe(
    attack_acts: np.ndarray,
    honest_acts: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean-difference probe direction.

    Args:
        attack_acts: shape (N_attack, d_model)
        honest_acts: shape (N_honest, d_model)

    Returns:
        direction: shape (d_model,), unit vector pointing honest -> attack
    """
    direction = attack_acts.mean(axis=0) - honest_acts.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError(
            "Probe direction norm is near zero. "
            "Attack and honest activations appear identical at this layer."
        )
    return direction / norm


def score_probe(direction: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """
    Project activations onto the probe direction.

    Args:
        direction: shape (d_model,)
        acts: shape (N, d_model)

    Returns:
        scores: shape (N,), higher = more attack-like
    """
    return acts @ direction


def evaluate_probe(
    direction: np.ndarray,
    test_attack_acts: np.ndarray,
    test_honest_acts: np.ndarray,
) -> float:
    """
    Compute AUROC of the probe on a test set.

    Returns:
        auroc: float in [0, 1]. 0.5 = chance, 1.0 = perfect.
    """
    acts = np.concatenate([test_attack_acts, test_honest_acts], axis=0)
    labels = np.concatenate([
        np.ones(len(test_attack_acts)),
        np.zeros(len(test_honest_acts)),
    ])
    scores = score_probe(direction, acts)
    return float(roc_auc_score(labels, scores))


def layer_sweep(
    train_attack: dict,
    train_honest: dict,
    test_attack: dict,
    test_honest: dict,
    layers: list,
) -> dict:
    """
    Train and evaluate a probe at every layer.

    Args:
        train_attack, train_honest: dicts mapping layer -> (N, d_model)
        test_attack, test_honest: same structure

    Returns:
        results: dict mapping layer -> {"direction": ..., "auroc": float}
    """
    results = {}
    for layer in layers:
        direction = train_probe(train_attack[layer], train_honest[layer])
        auroc = evaluate_probe(direction, test_attack[layer], test_honest[layer])
        results[layer] = {"direction": direction, "auroc": auroc}
        print(f"  Layer {layer:2d}: AUROC = {auroc:.4f}")
    return results


def best_layer(sweep_results: dict) -> Tuple[int, float]:
    """Return (layer_idx, auroc) for the best layer in a sweep."""
    best = max(sweep_results.items(), key=lambda x: x[1]["auroc"])
    return best[0], best[1]["auroc"]
