# The Internal Geometry of Strategic Reasoning in Thinking Language Models

Code and results for a mechanistic interpretability study of strategic reasoning in DeepSeek-R1-Distill-Qwen-14B. I extract difference-of-means steering vectors for strategic reasoning sub-behaviors from the model's residual stream, test their geometry and causal role, and check whether they transfer within the Qwen-14B model family.

I built the multi-label annotation, calibration by manually annotating and checking the definitions aligned or not, and controlled intervention pipeline after identifying weaknesses in an earlier mutually-exclusive annotation setup. All experiments were run on Google Colab (A100).

---

## Current status

This is an active research project. The results support a robust, causally active opponent-modeling-associated direction in the residual stream, but I do not claim this direction is a pure semantic feature or that steering improves strategic reasoning. The main finding is that the direction behaves non-atomically: positive steering increases opponent-modeling-labeled reasoning, while projection-out ablation paradoxically also increases opponent-modeling-labeled segments. I am still working on human evaluation of the steered outputs and cross-architecture replication.

This README should be read as a snapshot of an active research project rather than a finalized paper artifact; additional robustness checks and component-level analyses are still being added.

## What I do not claim

- I do not claim the vector is a pure opponent-modeling feature.
- I do not claim steering improves strategic reasoning.
- I have not identified a full circuit or specific attention heads.
- I do not claim broad cross-architecture generality beyond the Qwen-14B family.
- I treat the LLM annotations as operational behavioral labels, not as ground truth about what the model is internally doing.

---

## Setup

| Item | Choice |
|---|---|
| Source model | DeepSeek-R1-Distill-Qwen-14B (48 layers, hidden dim 5120) |
| Transfer model | Qwen-2.5-14B-Instruct (same Qwen-14B family, non-R1-distilled) |
| Main task set | 300 game-theoretic reasoning tasks |
| OOD tasks | 50 held-out strategic tasks |
| Intervention layer | 24 |
| Vector type | difference-of-means (with-vs-without centering) |
| Annotator | GPT-5.4, chain-mode |
| Segmentation | paragraph-split with 1200-char hard-wrap |

---

## Phase 0 — Annotation pipeline

The earlier version of this project used mutually exclusive labels with a "prefer strategic" priority rule. That made the geometry too easy to overinterpret because many reasoning segments naturally contain multiple behaviors at once.

I built a multi-label annotation pipeline with no priority hierarchy, an explicit `none_other` label, and per-label definitions with positive and negative examples. The 10-label taxonomy:

| Label | What it covers |
|---|---|
| `opponent_modeling` | Reasoning about what another agent believes, expects, or will do |
| `iterated_reasoning` | Recursive "I think they think..." reasoning |
| `equilibrium_identification` | Nash equilibria, dominant strategies, solution concepts |
| `payoff_analysis` | Comparing outcomes, utilities, costs |
| `strategic_uncertainty` | Uncertainty over another agent's action or strategy |
| `cooperative_reasoning` | Fairness, trust, coordination |
| `initialization` | Restating setup, listing options |
| `deduction` | Non-social logical derivation or calculation |
| `backtracking` | Revising a previous step |
| `none_other` | Does not fit other categories |

**Code:** `annotation/annotation_v2.py`, `annotation/annotate_chains.py`, `annotation/segment_chains.py`

---

## Phase 0.5 — Gold calibration

I hand-labeled 150 segments (stratified sample) and compared the LLM annotator against my labels.

| Metric | Value |
|---|---|
| Gold segments | 150 |
| GPT-5.4 chain-mode macro-F1 | 0.770 |
| GPT-5.2 per-segment macro-F1 | 0.739 |
| GPT-5.4-mini macro-F1 | 0.653 (failed — cooperative_reasoning collapse) |
| Weakest label | deduction (F1 ~0.55) |

Deduction is noisy. The annotator over-applies it roughly 2x relative to my gold labels. I keep this in mind for all downstream analysis involving the deduction direction.

**Code:** `annotation/sample_gold_segments.py`, `annotation/evaluate_gold.py`

---

## Phase 1 — Full annotation

After calibration, I annotated the full corpus of 300 R1 reasoning chains.

| Quantity | Value |
|---|---|
| Tasks | 300 |
| Total annotated segments | 14,645 |
| Think-region segments | 13,010 |
| Multi-label segments | 57.6% |

Label distribution (think-region only):

| Label | Rate |
|---|---|
| payoff_analysis | 53.2% |
| opponent_modeling | 30.7% |
| deduction | 22.6% |
| strategic_uncertainty | 12.9% |
| equilibrium_identification | 12.4% |
| backtracking | 12.4% |
| none_other | 10.4% |
| initialization | 10.0% |
| cooperative_reasoning | 6.5% |
| iterated_reasoning | 0.6% |

`iterated_reasoning` has only 80 think-region segments. I do not build claims on it.

Top co-occurrence: opponent_modeling + payoff_analysis (2,430 segments). opponent_modeling + deduction co-occurrence: 181 segments (1.4%).

**Code:** `annotation/annotate_chains.py`

---

## Phase 2 — Activation extraction and geometry

For each annotated segment, I extracted residual-stream activations across all 48 layers using mean pooling across token positions. Then for each label, I computed a difference-of-means direction:

```
u_c = mean(activations where segment has label c) - mean(activations where segment does not have label c)
```

This with-vs-without formulation is the correct one for multi-label data.

### Cosine between opponent-modeling and deduction directions

| Centering method | Layer 24 | All-layer mean | All layers negative? |
|---|---|---|---|
| With-vs-without (primary) | -0.707 | -0.723 | 48/48 |
| Leave-one-out | -0.707 | -0.723 | 48/48 |
| Class-balanced | -0.371 | -0.342 | 48/48 |

The class-balanced number is weaker. Both numbers are real; which one is more meaningful depends on how you weight the dominant payoff_analysis category.

### Statistical validation (with-vs-without)

| Test | Result |
|---|---|
| Permutation test (n=1000) | p < 0.001 (null mean = -0.005) |
| Bootstrap 95% CI (n=1000) | [-0.727, -0.682] |
| Split-half (n=100) | mean = -0.705, std = 0.011, 200/200 negative |

### Important caveat

All category means have cosine >= 0.94 with the global mean. The DoM vectors are small perturbations off a dominant shared "thinking" direction. The antagonism lives in roughly 5% of the representational variance.

### Pairwise raw cosine (no DoM subtraction, L24)

| Pair | Cosine |
|---|---|
| opp_mod vs deduction | +0.946 |
| opp_mod vs payoff_analysis | +0.985 |
| opp_mod vs strategic_uncertainty | +0.984 |

Raw representations are nearly identical. The antagonism only emerges after subtracting the global mean.

### Full cosine matrix (with-vs-without, L24)

```
             opp-mod  iter-reas  equil-id   payoff  strat-unc  coop-reas     init  deduction  backtrack  none/other
opp-mod       +1.000    +0.526    +0.111   +0.432     +0.575     +0.327   -0.153     -0.707     +0.306      -0.610
iter-reas     +0.526    +1.000    +0.172   -0.051     +0.483     +0.569   +0.191     -0.407     +0.170      -0.243
equil-id      +0.111    +0.172    +1.000   -0.291     +0.215     +0.230   -0.169     -0.206     +0.224      -0.011
payoff        +0.432    -0.051    -0.291   +1.000     -0.157     -0.136   -0.349     -0.186     -0.023      -0.719
strat-unc     +0.575    +0.483    +0.215   -0.157     +1.000     +0.462   +0.020     -0.463     +0.408      -0.228
coop-reas     +0.327    +0.569    +0.230   -0.136     +0.462     +1.000   +0.382     -0.401     +0.101      -0.162
init          -0.153    +0.191    -0.169   -0.349     +0.020     +0.382   +1.000     -0.144     -0.338      +0.138
deduction     -0.707    -0.407    -0.206   -0.186     -0.463     -0.401   -0.144     +1.000     -0.366      +0.110
backtrack     +0.306    +0.170    +0.224   -0.023     +0.408     +0.101   -0.338     -0.366     +1.000      +0.018
none/other    -0.610    -0.243    -0.011   -0.719     -0.228     -0.162   +0.138     +0.110     +0.018      +1.000
```

### SVD of category-mean matrix (L24)

| Component | Explained variance | Cumulative |
|---|---|---|
| SV1 | 34.3% | 34.3% |
| SV2 | 26.5% | 60.8% |
| SV3 | 14.9% | 75.8% |

No single component dominates. The geometry needs 3+ axes.

**Code:** `scripts/phase2_geometry.py`

---

## Phase 2.5 — Single-axis disambiguation

I ran three tests to check whether the opp-mod/deduction antagonism is a single signed axis or something more complex.

### Test 1: Linear probe vs DoM direction

| Diagnostic | Result |
|---|---|
| Probe CV accuracy (5-fold) | 0.87 |
| cos(probe weight, DoM contrast) | +0.146 |

The probe classifies well but finds a nearly orthogonal direction to the DoM contrast. The probe weight is 97.8% orthogonal to the top-6 SVD subspace of the DoM category-mean matrix — it uses within-class variance structure that means-based analysis does not see.

### Regularization sweep (L24)

| C (regularization) | CV accuracy | cos(w, DoM) |
|---|---|---|
| 0.0001 (strongest) | 0.90 | +0.39 |
| 1.0 (default) | 0.87 | +0.15 |
| 100.0 (weakest) | 0.86 | +0.13 |

Strong regularization pushes the probe toward the DoM direction and improves accuracy. The DoM direction is a high-SNR direction, but the unregularized probe finds higher-dimensional features.

### Test 2: SVD depth profile

SV1 never exceeds 58% at any layer (mean 37.8%). 0/48 layers have SV1 > 70%.

### Test 3: Co-occurrence geometry (181 BOTH segments)

BOTH segments project at +7.9 on u_opp and -5.3 on u_ded. Relative position on contrast axis: +0.636 (leans toward opp_mod). Not consistent with single-axis cancellation.

### Within-class variance

Deduction has 1.49x more total variance than opponent-modeling. Fisher discriminant ratio: probe direction is 2.53x better than DoM. Deduction std along DoM axis: 12.37 (vs opp-mod: 6.62).

### Verdict

All three tests agree: the geometry is multi-dimensional. The antagonism is real but coexists with independent structure.

**Code:** `scripts/phase2_5_probe_svd.py`, `scripts/phase2_5_extended.py`

---

## Phase 3 — R1 in-distribution interventions (supplementary)

I ran steering and projection-out ablation on R1-Distill using 50 in-distribution strategic tasks. 8 conditions. All outputs re-annotated with GPT-5.4 chain-mode. These are supplementary to the OOD results in Phase 4A.

### Ablation results

| Condition | opp_mod | Δ from baseline | deduction |
|---|---|---|---|
| baseline | 33.5% | — | 19.1% |
| ablate_opp | 45.0% | +11.6pp | 18.0% |
| ablate_random | 31.5% | -2.0pp | 19.8% |
| ablate_payoff | 32.7% | -0.8pp | 21.2% |
| ablate_probe | 34.0% | +0.5pp | 16.7% |

### Steering results

| alpha | opp_mod | deduction | strat_unc | segments |
|---|---|---|---|---|
| -0.5 | 0.1% | 13.5% | 1.2% | 6257 |
| 0.0 | 33.5% | 19.1% | 17.3% | 2540 |
| +0.2 | 41.5% | 14.3% | 26.1% | 2273 |
| +0.3 | 47.4% | 15.5% | 27.1% | 2442 |

Phase 3 used max_new_tokens=3072. 30-46% of outputs were truncated. Relative comparisons are valid within-regime.

**Code:** `scripts/phase3_ablation_steering.py`, `scripts/phase3_annotation.py`

---

## Phase 4A — R1-Distill on OOD tasks (primary results)

Same 8 conditions as Phase 3, run on 50 held-out OOD strategic tasks. max_new_tokens=6144.

### Ablation results

| Condition | opp_mod | Δ from baseline | deduction | segments |
|---|---|---|---|---|
| baseline | 23.9% | — | 22.6% | 4029 |
| ablate_opp | 32.0% | +8.1pp | 22.2% | 2632 |
| ablate_random | 25.1% | +1.3pp | 23.5% | 3535 |
| ablate_payoff | 24.6% | +0.7pp | 18.7% | 3135 |
| ablate_probe | 23.5% | -0.4pp | 24.5% | 3719 |

The ablation paradox replicates out of distribution.

### Steering results

| alpha | opp_mod | deduction | strat_unc | segments |
|---|---|---|---|---|
| -0.5 | 0.0% | 9.1% | 0.7% | 10741 |
| 0.0 | 23.9% | 22.6% | 15.8% | 4029 |
| +0.2 | 38.9% | 20.0% | 14.0% | 3911 |
| +0.3 | 44.2% | 7.4% | 16.2% | 4911 |

At alpha=-0.5, the model spirals (10,741 segments, 80% truncated at 6144 tokens).

**Code:** `scripts/phase4_generation.py`, `scripts/phase4_annotation.py`

---

## Phase 4B — OOD and Same-Family Base-Model Transfer

Same R1-derived vectors applied to the base model. 5 conditions.

### Results

| Condition | opp_mod | deduction | payoff | segments |
|---|---|---|---|---|
| baseline | 28.1% | 18.6% | 63.0% | 652 |
| ablate_opp | 41.0% | 17.2% | 66.7% | 534 |
| ablate_random | 27.0% | 24.1% | 62.4% | 611 |
| steer_+0.2 | 41.5% | 20.7% | 65.1% | 479 |
| steer_+0.3 | 39.7% | 27.7% | 73.1% | 4049 |

The ablation paradox appears on the base model (+12.9pp, specific to u_opp). At steer_+0.3, the base model spirals (4049 segments, 6.2x baseline). Unlike R1, deduction goes up and backtracking stays at 0.0% across all conditions.

Base model sample sizes are small (479-652 segments for non-spiraling conditions).

---

## Cross-setting summary

### Ablation paradox

| Setting | baseline | ablate_opp | Δ_opp | ablate_random | Δ_rand |
|---|---|---|---|---|---|
| Phase 3 (R1, in-dist) | 33.5% | 45.0% | +11.6pp | 31.5% | -2.0pp |
| Phase 4A (R1, OOD) | 23.9% | 32.0% | +8.1pp | 25.1% | +1.3pp |
| Phase 4B (Base, OOD) | 28.1% | 41.0% | +12.9pp | 27.0% | -1.1pp |

### Steering dose-response (opp_mod)

| alpha | P3 (R1 in-dist) | P4A (R1 OOD) | P4B (Base OOD) |
|---|---|---|---|
| -0.5 | 0.1% | 0.0% | — |
| 0.0 | 33.5% | 23.9% | 28.1% |
| +0.2 | 41.5% | 38.9% | 41.5% |
| +0.3 | 47.4% | 44.2% | 39.7% |

---

## Interpretation

The opponent-modeling DoM direction has asymmetric effects depending on how it is applied:

- **Positive steering** increases opponent-modeling-labeled segments and suppresses deduction on R1.
- **Strong negative steering** suppresses opponent-modeling but causes degeneration.
- **Projection-out ablation** paradoxically increases opponent-modeling-labeled segments, possibly by disrupting transition dynamics between reasoning modes.

One candidate explanation is that the direction participates in transitions into or out of opponent-modeling-heavy reasoning modes. I consider this plausible but not established.

The ablation paradox appearing on the base model suggests that the relevant representational structure is at least partly preserved within the Qwen-14B family, rather than being unique to R1-style reasoning distillation. What reasoning training does appear to install is the behavioral coupling (opp-mod up / deduction down under steering) and the self-correction vocabulary (backtracking).

---

## Limitations

- **Single model family.** Everything here is Qwen-14B. I have not tested Llama, Mistral, or anything else yet.
- **Single annotator.** GPT-5.4 only. No second judge. Relative comparisons are more meaningful than absolute rates because the same annotator/prompt is used across conditions, but condition-dependent annotation bias remains possible.
- **Deduction labels are noisy.** F1 ~0.55 at gold calibration.
- **No human evaluation of steered/ablated outputs.** Planned but not done.
- **Class-balanced centering gives -0.37** vs -0.71 for with-vs-without.
- **Base model sample sizes are small** (479-652 segments for non-spiraling conditions).
- **No circuit-level localization.** Single-layer intervention only.
- **Interventions do not improve reasoning.** High-alpha steering is pathological.

---

## Repo structure

```
data/
  final_dataset.json           # 300 training tasks
  ood.json                     # 50 OOD tasks
  r1_qwen14b_chains.json       # 300 reasoning chains

annotation/
  annotation_v2.py             # annotation prompt + core logic
  annotate_chains.py           # chain-mode annotation runner
  segment_chains.py            # paragraph segmenter
  sample_gold_segments.py      # stratified gold sampler
  evaluate_gold.py             # F1 evaluator

scripts/
  phase2_geometry.py           # activation extraction + DoM + robustness suite
  phase2_5_probe_svd.py        # probe vs DoM, SVD depth, co-occurrence
  phase2_5_extended.py         # multi-layer probes, regularization sweep, variance
  phase3_ablation_steering.py  # R1 in-dist generation (8 conditions)
  phase3_annotation.py         # segment + annotate Phase 3 outputs
  phase4_generation.py         # R1 OOD + base OOD generation
  phase4_annotation.py         # segment + annotate Phase 4 outputs

results/ ( this will be added in future revisions )
  phase2_geometry_report.json
  phase25_results.json
  phase25_extended_results.json
  phase3_annotation_report.json
  phase4_annotation_report.json
```

Large files (activations, raw outputs) are not committed. They can be regenerated from the scripts.

---

