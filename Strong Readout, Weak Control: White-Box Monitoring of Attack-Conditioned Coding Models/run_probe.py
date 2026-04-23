"""
run_probe.py

Trains the mean-difference probe on 80% of data and evaluates across
all three conditions (honest, attack, defensive) at each layer.

Experiments:
  1. Layer sweep          -- attack vs honest AUROC (80/20 split)
  2. Confounder test      -- defensive vs honest AUROC (same probe)
  3. Full-dataset AUROC   -- no split, for comparison
  4. Logistic regression  -- upper bound on linear separability
  5. PCA analysis         -- do top PCs capture condition variance?
  6. Between-class variance ratio -- layer-wise signal strength
  7. Reasoning correlation -- do examples with explicit backdoor plans
                              in <think> score higher?
  8. Null baselines        -- random direction and shuffled labels

Output:
  data/arm_a_final/results/
    results.json
    layer_sweep.png
    score_distributions.png
    variance_ratio.png

Usage:
    python run_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

ACTIVATIONS_DIR = Path("data/arm_a_final/activations")
RESULTS_DIR     = Path("data/arm_a_final/results")
METADATA_PATH   = ACTIVATIONS_DIR / "metadata.json"

PROBE_LAYERS = [8, 14, 20, 26, 32, 38]
CONDITIONS   = ["honest", "attack", "defensive"]


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_activations() -> dict:
    acts = {}
    for cond in CONDITIONS:
        acts[cond] = {}
        for layer in PROBE_LAYERS:
            path = ACTIVATIONS_DIR / f"{cond}_layer{layer}.npy"
            arr = np.load(path)
            assert arr.shape == (125, 5120), f"Unexpected shape {arr.shape} in {path}"
            acts[cond][layer] = arr.astype(np.float32)
    print("Activations loaded.")
    return acts


def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Probe utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_probe_direction(attack_acts: np.ndarray, honest_acts: np.ndarray) -> np.ndarray:
    direction = attack_acts.mean(axis=0) - honest_acts.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Probe direction has near-zero norm.")
    return direction / norm


def probe_scores(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return acts @ direction


def auroc(pos_acts: np.ndarray, neg_acts: np.ndarray, direction: np.ndarray) -> float:
    scores = np.concatenate([probe_scores(pos_acts, direction), probe_scores(neg_acts, direction)])
    labels = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
    return float(roc_auc_score(labels, scores))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1+2: Layer sweep (80/20 split)
# ─────────────────────────────────────────────────────────────────────────────

def run_layer_sweep(acts: dict) -> dict:
    print("\n=== Experiment 1+2: Layer Sweep (80/20 split) ===")
    results = {"attack_vs_honest": {}, "defensive_vs_honest": {}, "attack_vs_defensive": {}}

    for layer in PROBE_LAYERS:
        attack  = acts["attack"][layer]
        honest  = acts["honest"][layer]
        defense = acts["defensive"][layer]

        n = len(attack)
        split = int(0.8 * n)

        direction = compute_probe_direction(attack[:split], honest[:split])

        results["attack_vs_honest"][layer]    = auroc(attack[split:],  honest[split:],  direction)
        results["defensive_vs_honest"][layer] = auroc(defense[split:], honest[split:],  direction)
        results["attack_vs_defensive"][layer] = auroc(attack[split:],  defense[split:], direction)

        print(
            f"  Layer {layer:2d}: "
            f"att/hon={results['attack_vs_honest'][layer]:.3f}  "
            f"def/hon={results['defensive_vs_honest'][layer]:.3f}  "
            f"att/def={results['attack_vs_defensive'][layer]:.3f}"
        )

    best = max(PROBE_LAYERS, key=lambda l: results["attack_vs_honest"][l])
    results["best_layer"] = best
    print(f"\n  Best layer: {best}  (AUROC {results['attack_vs_honest'][best]:.3f})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Full-dataset AUROC (no split)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_auroc(acts: dict) -> dict:
    print("\n=== Experiment 3: Full-Dataset AUROC (no split) ===")
    results = {}

    for layer in PROBE_LAYERS:
        direction = compute_probe_direction(acts["attack"][layer], acts["honest"][layer])
        results[layer] = {
            "attack_vs_honest":    auroc(acts["attack"][layer],    acts["honest"][layer],    direction),
            "defensive_vs_honest": auroc(acts["defensive"][layer], acts["honest"][layer],    direction),
            "attack_vs_defensive": auroc(acts["attack"][layer],    acts["defensive"][layer], direction),
        }
        print(
            f"  Layer {layer:2d}: "
            f"att/hon={results[layer]['attack_vs_honest']:.3f}  "
            f"def/hon={results[layer]['defensive_vs_honest']:.3f}  "
            f"att/def={results[layer]['attack_vs_defensive']:.3f}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Logistic regression upper bound
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(acts: dict) -> dict:
    """
    NOTE: In this regime (~100 training examples vs 5120 dimensions) the
    classifier is heavily overparameterised and typically achieves AUROC=1.0
    in-sample. This is NOT evidence of a meaningful generalizable feature --
    it just shows the classes are linearly separable in-sample.
    """
    print("\n=== Experiment 4: Logistic Regression Upper Bound ===")
    print("  (WARNING: overparameterised regime -- in-sample AUROC is not meaningful)")
    results = {}

    for layer in PROBE_LAYERS:
        X = np.concatenate([acts["attack"][layer], acts["honest"][layer]])
        y = np.concatenate([np.ones(125), np.zeros(125)])

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        scores = clf.predict_proba(X)[:, 1]
        results[layer] = float(roc_auc_score(y, scores))
        print(f"  Layer {layer:2d}: AUROC = {results[layer]:.3f}  (in-sample)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5: PCA analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_analysis(acts: dict) -> dict:
    print("\n=== Experiment 5: PCA Analysis ===")
    results = {}

    for layer in PROBE_LAYERS:
        X = np.concatenate([
            acts["attack"][layer],
            acts["honest"][layer],
            acts["defensive"][layer],
        ])
        labels = np.array(["attack"] * 125 + ["honest"] * 125 + ["defensive"] * 125)

        pca = PCA(n_components=10)
        pca.fit(X)
        explained = pca.explained_variance_ratio_

        X_pca = pca.transform(X)
        attack_mean  = X_pca[:125].mean(axis=0)
        honest_mean  = X_pca[125:250].mean(axis=0)
        defense_mean = X_pca[250:].mean(axis=0)

        results[layer] = {
            "explained_variance_top10": explained.tolist(),
            "pc1_attack_mean":   float(attack_mean[0]),
            "pc1_honest_mean":   float(honest_mean[0]),
            "pc1_defense_mean":  float(defense_mean[0]),
        }
        print(f"  Layer {layer:2d}: top-3 explained var: {explained[:3]}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 6: Between-class variance ratio
# ─────────────────────────────────────────────────────────────────────────────

def run_variance_ratio(acts: dict) -> dict:
    print("\n=== Experiment 6: Between-Class Variance Ratio ===")
    results = {}

    for layer in PROBE_LAYERS:
        direction = compute_probe_direction(acts["attack"][layer], acts["honest"][layer])
        attack_scores  = probe_scores(acts["attack"][layer],    direction)
        honest_scores  = probe_scores(acts["honest"][layer],    direction)
        defense_scores = probe_scores(acts["defensive"][layer], direction)

        all_scores = np.concatenate([attack_scores, honest_scores, defense_scores])
        total_var  = float(np.var(all_scores))

        grand_mean   = all_scores.mean()
        between_var  = float(np.array([
            125 * (attack_scores.mean()  - grand_mean)**2,
            125 * (honest_scores.mean()  - grand_mean)**2,
            125 * (defense_scores.mean() - grand_mean)**2,
        ]).sum() / (375 - 1))

        ratio = between_var / (total_var + 1e-10)
        results[layer] = {"between_class_variance": between_var, "total_variance": total_var, "ratio": ratio}
        print(f"  Layer {layer:2d}: ratio = {ratio:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 7: Reasoning correlation
# ─────────────────────────────────────────────────────────────────────────────

def run_reasoning_correlation(acts: dict, metadata: dict, best_layer: int) -> dict:
    print(f"\n=== Experiment 7: Reasoning Correlation (layer {best_layer}) ===")

    attack_meta = metadata.get("attack", [])
    direction = compute_probe_direction(acts["attack"][best_layer], acts["honest"][best_layer])
    scores    = probe_scores(acts["attack"][best_layer], direction)

    with_plan    = [s for s, m in zip(scores, attack_meta) if m.get("reasoning_plans_backdoor")]
    without_plan = [s for s, m in zip(scores, attack_meta) if not m.get("reasoning_plans_backdoor")]

    result = {
        "n_with_plan":    len(with_plan),
        "n_without_plan": len(without_plan),
        "mean_with_plan":    float(np.mean(with_plan))    if with_plan    else None,
        "mean_without_plan": float(np.mean(without_plan)) if without_plan else None,
    }

    print(f"  With explicit backdoor plan in <think>:    n={result['n_with_plan']:3d}  "
          f"mean_score={result['mean_with_plan']}")
    print(f"  Without explicit backdoor plan in <think>: n={result['n_without_plan']:3d}  "
          f"mean_score={result['mean_without_plan']}")
    print(f"  Note: 'with plan' examples had LOWER scores at several layers in our run.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 8: Null baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_null_baselines(acts: dict) -> dict:
    print("\n=== Experiment 8: Null Baselines ===")
    rng = np.random.RandomState(42)
    results = {}

    for layer in PROBE_LAYERS:
        d = acts["attack"][layer].shape[1]
        random_dir = rng.randn(d).astype(np.float32)
        random_dir /= np.linalg.norm(random_dir)

        random_auroc = auroc(acts["attack"][layer], acts["honest"][layer], random_dir)

        all_acts   = np.concatenate([acts["attack"][layer], acts["honest"][layer]])
        labels     = np.concatenate([np.ones(125), np.zeros(125)])
        rng.shuffle(labels)
        direction  = compute_probe_direction(acts["attack"][layer], acts["honest"][layer])
        scores     = probe_scores(all_acts, direction)
        shuffled_auroc = float(roc_auc_score(labels, scores))

        results[layer] = {
            "random_direction_auroc": round(random_auroc, 4),
            "shuffled_labels_auroc":  round(shuffled_auroc, 4),
        }
        print(f"  Layer {layer:2d}: random_dir={random_auroc:.3f}  shuffled_labels={shuffled_auroc:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(layer_sweep_results, variance_ratio, full_auroc, acts, out_dir):
    layers = PROBE_LAYERS

    # Layer sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, [layer_sweep_results["attack_vs_honest"][l]    for l in layers], "o-", label="Attack vs Honest")
    ax.plot(layers, [layer_sweep_results["defensive_vs_honest"][l] for l in layers], "s--", label="Defensive vs Honest")
    ax.plot(layers, [layer_sweep_results["attack_vs_defensive"][l] for l in layers], "^:", label="Attack vs Defensive")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (80/20 split)")
    ax.set_title("Mean-Difference Probe: Layer Sweep")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_sweep.png", dpi=150)
    plt.close(fig)

    # Score distributions at best layer
    best_l = layer_sweep_results["best_layer"]
    direction = compute_probe_direction(acts["attack"][best_l], acts["honest"][best_l])

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, color in [("honest", "steelblue"), ("attack", "firebrick"), ("defensive", "orange")]:
        scores = probe_scores(acts[cond][best_l], direction)
        ax.hist(scores, bins=30, alpha=0.6, color=color, label=cond, density=True)
    ax.set_xlabel("Probe Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Probe Score Distributions (Layer {best_l})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    acts     = load_activations()
    metadata = load_metadata()

    layer_sweep    = run_layer_sweep(acts)
    full_auroc     = run_full_auroc(acts)
    lr_results     = run_logistic_regression(acts)
    pca_results    = run_pca_analysis(acts)
    variance_ratio = run_variance_ratio(acts)
    null_baselines = run_null_baselines(acts)

    best_layer     = layer_sweep["best_layer"]
    reasoning_corr = run_reasoning_correlation(acts, metadata, best_layer)

    all_results = {
        "layer_sweep":           layer_sweep,
        "full_auroc":            {str(k): v for k, v in full_auroc.items()},
        "logistic_regression":   {str(k): v for k, v in lr_results.items()},
        "pca_analysis":          {str(k): v for k, v in pca_results.items()},
        "variance_ratio":        {str(k): v for k, v in variance_ratio.items()},
        "null_baselines":        {str(k): v for k, v in null_baselines.items()},
        "reasoning_correlation": reasoning_corr,
        "best_layer":            best_layer,
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    make_plots(layer_sweep, variance_ratio, full_auroc, acts, RESULTS_DIR)

    # Print interpretation
    best_ah = full_auroc[best_layer]["attack_vs_honest"]
    best_dh = full_auroc[best_layer]["defensive_vs_honest"]

    print("\n" + "=" * 56)
    print("  RESULTS SUMMARY")
    print("=" * 56)
    print(f"\n  Best layer: {best_layer}")
    print(f"  Attack vs Honest AUROC (full):    {best_ah:.3f}")
    print(f"  Defensive vs Honest AUROC (full): {best_dh:.3f}")

    if best_ah > 0.75 and best_dh < 0.65:
        print("\n  Interpretation: strong attack signal with low confounder (clean result)")
    elif best_ah > 0.75 and best_dh > 0.65:
        print("\n  Interpretation: probe partially confounded by defensive prompts")
        print("  The direction is NOT a pure malicious-intent feature.")
    else:
        print(f"\n  Interpretation: weak or null attack signal (AUROC={best_ah:.3f})")

    print(f"\n  Output: {RESULTS_DIR}/")
    print("  Next: python run_behavioral_monitor.py")


if __name__ == "__main__":
    main()
