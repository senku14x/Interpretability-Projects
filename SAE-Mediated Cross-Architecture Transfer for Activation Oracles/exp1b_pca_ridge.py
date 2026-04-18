"""
Exp 1b: Improved raw cross-architecture baseline
================================================

Uses:
  standardize -> PCA -> ridge regression -> reconstruct -> Gemma oracle

This is a separate script. It does NOT modify exp_sae_transfer.py.
It imports your existing exp_sae_transfer.py for shared config/helpers.

Run:
  cd /content/activation_oracles
  python exp1b_pca_ridge.py

Or from a notebook:
  import importlib.util
  spec = importlib.util.spec_from_file_location("exp1b_pca_ridge", "/content/activation_oracles/exp1b_pca_ridge.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  mod.run_exp1b_pca_ridge()
"""

import os
import gc
import json
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Import your existing script as a module, without modifying it
# ---------------------------------------------------------------------
import importlib.util

BASE_PATH = "/content/activation_oracles/exp_sae_transfer.py"
spec = importlib.util.spec_from_file_location("exp_sae_transfer", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# ---------------------------------------------------------------------
# New config for the improved raw baseline
# ---------------------------------------------------------------------
PROJ_PCA_RANK = 256
PROJ_RIDGE_ALPHA = 1.0
RESULTS_FILE = os.path.join(base.RESULTS_DIR, "exp1b_pca_ridge_results.json")


# ---------------------------------------------------------------------
# PCA + Ridge helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def fit_pca_ridge_projection(
    X: torch.Tensor,   # [N, d_src]
    Y: torch.Tensor,   # [N, d_tgt]
    rank: int = 256,
    alpha: float = 1.0,
):
    """
    Train pipeline:
      1. standardize source and target separately
      2. PCA on source and target train sets
      3. ridge regression in reduced space
      4. reconstruct back to target hidden space
    """
    X = X.float()
    Y = Y.float()

    # Train statistics only
    mu_x = X.mean(dim=0, keepdim=True)
    std_x = X.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    mu_y = Y.mean(dim=0, keepdim=True)
    std_y = Y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)

    Xz = (X - mu_x) / std_x
    Yz = (Y - mu_y) / std_y

    # Effective rank
    eff_rank = min(rank, Xz.shape[0] - 1, Xz.shape[1], Yz.shape[0] - 1, Yz.shape[1])
    if eff_rank < 2:
        raise ValueError(f"Effective PCA rank too small: {eff_rank}")

    # PCA on standardized activations
    _, _, Vx = torch.pca_lowrank(Xz, q=eff_rank, center=False)
    _, _, Vy = torch.pca_lowrank(Yz, q=eff_rank, center=False)

    Bx = Vx[:, :eff_rank]   # [d_src, k]
    By = Vy[:, :eff_rank]   # [d_tgt, k]

    Xk = Xz @ Bx            # [N, k]
    Yk = Yz @ By            # [N, k]

    # Ridge in reduced space
    I = torch.eye(eff_rank, device=X.device, dtype=X.dtype)
    XtX = Xk.T @ Xk
    XtY = Xk.T @ Yk
    W = torch.linalg.solve(XtX + alpha * I, XtY)   # [k, k]

    # Diagnostics on train data
    Yk_pred = Xk @ W
    Yz_pred = Yk_pred @ By.T
    Y_pred = Yz_pred * std_y + mu_y

    reduced_cos = nn.functional.cosine_similarity(Yk, Yk_pred, dim=-1).mean().item()
    hidden_cos = nn.functional.cosine_similarity(Y, Y_pred, dim=-1).mean().item()
    hidden_mse = ((Y - Y_pred) ** 2).mean().item()

    proj = {
        "mu_x": mu_x,
        "std_x": std_x,
        "mu_y": mu_y,
        "std_y": std_y,
        "Bx": Bx,
        "By": By,
        "W": W,
        "rank": eff_rank,
        "alpha": alpha,
    }

    stats = {
        "rank": eff_rank,
        "alpha": alpha,
        "train_reduced_cosine": reduced_cos,
        "train_hidden_cosine": hidden_cos,
        "train_hidden_mse": hidden_mse,
    }
    return proj, stats


@torch.no_grad()
def apply_pca_ridge_projection(proj: dict, vecs_KD: torch.Tensor) -> torch.Tensor:
    """
    Apply trained PCA+ridge projection to [K, d_src] vectors.
    Returns [K, d_tgt] in Gemma hidden space.
    """
    X = vecs_KD.to(proj["mu_x"].device).float()

    Xz = (X - proj["mu_x"]) / proj["std_x"]
    Xk = Xz @ proj["Bx"]
    Yk = Xk @ proj["W"]
    Yz = Yk @ proj["By"].T
    Y = Yz * proj["std_y"] + proj["mu_y"]

    return Y.to(base.DTYPE).cpu()


# ---------------------------------------------------------------------
# Main improved experiment
# ---------------------------------------------------------------------
def run_exp1b_pca_ridge(rank: int = PROJ_PCA_RANK, alpha: float = PROJ_RIDGE_ALPHA):
    print("\n" + "=" * 70)
    print("EXP 1b: Raw Cross-Architecture Projection (PCA + Ridge)")
    print("=" * 70)
    base.set_all_seeds(base.SEED)

    gemma_layer = base.layer_percent_to_layer(base.GEMMA_MODEL, base.LAYER_PERCENT)
    llama_layer = base.layer_percent_to_layer(base.LLAMA_MODEL, base.LAYER_PERCENT)
    print(f"Gemma layer: {gemma_layer}, Llama layer: {llama_layer}")
    print(f"PCA rank: {rank}, ridge alpha: {alpha}")

    # --------------------------------------------------
    # Phase 1: Load / ensure caches
    # --------------------------------------------------
    print("\nPhase 1a: Gemma data...")
    gemma_model = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    gemma_eval = base.load_eval_data(base.GEMMA_MODEL, base.EVAL_DATASETS, model=gemma_model)
    gemma_proj_train = base.load_proj_train_data(base.GEMMA_MODEL, base.PROJ_TRAIN_DATASETS, model=gemma_model)

    llama_needs_extraction = False
    for ds_name in list(base.EVAL_DATASETS) + list(base.PROJ_TRAIN_DATASETS):
        num_train = base.PROJ_TRAIN_DATASETS.get(ds_name, {}).get("num_train", 0)
        num_test = base.EVAL_DATASETS.get(ds_name, {}).get("num_test", 0)
        for split, n in [("test", num_test), ("train", num_train)]:
            if n == 0:
                continue
            cfg = base._make_cls_config(ds_name, num_train, num_test, [split], base.LLAMA_MODEL)
            loader = base.ClassificationDatasetLoader(dataset_config=cfg)
            path = os.path.join(loader.dataset_config.dataset_folder, loader.get_dataset_filename(split))
            if not os.path.exists(path):
                llama_needs_extraction = True
                break

    if llama_needs_extraction:
        print("\nPhase 1b: Llama caches missing. Freeing Gemma, extracting Llama acts...")
        del gemma_model
        torch.cuda.empty_cache()
        gc.collect()

        llama_model = base.load_model_patched(base.LLAMA_MODEL, base.DTYPE)
        base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS, model=llama_model)
        base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS, model=llama_model)
        del llama_model
        torch.cuda.empty_cache()
        gc.collect()

        print("\nReloading Gemma...")
        gemma_model = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    else:
        print("\nPhase 1b: All Llama caches found ✓")

    llama_eval = base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS)
    llama_proj_train = base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS)

    # --------------------------------------------------
    # Phase 2: Alignment checks
    # --------------------------------------------------
    print("\nPhase 2: Asserting Gemma/Llama alignment...")
    for ds_name in base.EVAL_DATASETS:
        base.assert_aligned(gemma_eval[ds_name], llama_eval[ds_name], f"{ds_name}_test")
    for ds_name in base.PROJ_TRAIN_DATASETS:
        base.assert_aligned(gemma_proj_train[ds_name], llama_proj_train[ds_name], f"{ds_name}_train")

    # --------------------------------------------------
    # Phase 3: Fit improved raw baseline
    # --------------------------------------------------
    print("\nPhase 3: Fitting PCA+ridge projection on train split...")

    gemma_train_vecs = torch.cat(
        [torch.cat([dp.steering_vectors for dp in gemma_proj_train[ds]], dim=0)
         for ds in base.PROJ_TRAIN_DATASETS],
        dim=0,
    ).to(base.device)

    llama_train_vecs = torch.cat(
        [torch.cat([dp.steering_vectors for dp in llama_proj_train[ds]], dim=0)
         for ds in base.PROJ_TRAIN_DATASETS],
        dim=0,
    ).to(base.device)

    print(f"  Projection training: {gemma_train_vecs.shape[0]} vectors")
    print(f"  Gemma dim: {gemma_train_vecs.shape[1]}, Llama dim: {llama_train_vecs.shape[1]}")

    proj_aligned, aligned_stats = fit_pca_ridge_projection(
        llama_train_vecs,
        gemma_train_vecs,
        rank=rank,
        alpha=alpha,
    )
    print(f"  Aligned projection: {aligned_stats}")

    perm = torch.randperm(gemma_train_vecs.shape[0], device=gemma_train_vecs.device)
    proj_shuffled, shuffled_stats = fit_pca_ridge_projection(
        llama_train_vecs,
        gemma_train_vecs[perm],
        rank=rank,
        alpha=alpha,
    )
    print(f"  Shuffled projection: {shuffled_stats}")

    eval_geom = {}
    for ds_name in base.EVAL_DATASETS:
        g_vecs = torch.cat([dp.steering_vectors for dp in gemma_eval[ds_name]], dim=0).to(base.device)
        l_vecs = torch.cat([dp.steering_vectors for dp in llama_eval[ds_name]], dim=0).to(base.device)

        pred_aligned = apply_pca_ridge_projection(proj_aligned, l_vecs)
        pred_shuffled = apply_pca_ridge_projection(proj_shuffled, l_vecs)

        cos_a = nn.functional.cosine_similarity(g_vecs.cpu(), pred_aligned, dim=-1).mean().item()
        cos_s = nn.functional.cosine_similarity(g_vecs.cpu(), pred_shuffled, dim=-1).mean().item()
        mse_a = ((g_vecs.cpu() - pred_aligned) ** 2).mean().item()
        mse_s = ((g_vecs.cpu() - pred_shuffled) ** 2).mean().item()

        eval_geom[ds_name] = {
            "aligned_cos": cos_a,
            "shuffled_cos": cos_s,
            "aligned_mse": mse_a,
            "shuffled_mse": mse_s,
        }

        print(
            f"  {ds_name} eval: "
            f"aligned_cos={cos_a:.4f}, shuffled_cos={cos_s:.4f}, "
            f"aligned_mse={mse_a:.4f}, shuffled_mse={mse_s:.4f}"
        )

    # --------------------------------------------------
    # Phase 4: Oracle eval
    # --------------------------------------------------
    print("\nPhase 4: Oracle evaluation...")
    gemma_tokenizer = base.load_tokenizer(base.GEMMA_MODEL)
    gemma_model.add_adapter(base.LoraConfig(), adapter_name="default")
    submodule = base.get_hf_submodule(gemma_model, base.INJECTION_LAYER)

    all_results = {
        "projection_config": {
            "rank": rank,
            "alpha": alpha,
        },
        "projection_train_stats": {
            "aligned": aligned_stats,
            "shuffled": shuffled_stats,
        },
        "projection_eval_geometry": eval_geom,
        "datasets": {},
    }

    for ds_name in base.EVAL_DATASETS:
        print(f"\n{'='*50}\nDataset: {ds_name}\n{'='*50}")
        g_data = gemma_eval[ds_name]
        l_data = llama_eval[ds_name]
        ds_results = {}

        ds_results["C1_genuine"] = base.run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, g_data,
            lora_path=base.GEMMA_ORACLE_ADAPTER, label=f"C1_{ds_name}",
        )

        aligned_data = []
        for g_dp, l_dp in zip(g_data, l_data):
            dp_new = g_dp.model_copy(deep=True)
            dp_new.steering_vectors = apply_pca_ridge_projection(proj_aligned, l_dp.steering_vectors)
            aligned_data.append(dp_new)
        ds_results["C2_aligned"] = base.run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, aligned_data,
            lora_path=base.GEMMA_ORACLE_ADAPTER, label=f"C2_aligned_{ds_name}",
        )

        shuffled_data = []
        for g_dp, l_dp in zip(g_data, l_data):
            dp_new = g_dp.model_copy(deep=True)
            dp_new.steering_vectors = apply_pca_ridge_projection(proj_shuffled, l_dp.steering_vectors)
            shuffled_data.append(dp_new)
        ds_results["C2_shuffled"] = base.run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, shuffled_data,
            lora_path=base.GEMMA_ORACLE_ADAPTER, label=f"C2_shuffled_{ds_name}",
        )

        c1 = ds_results["C1_genuine"]["accuracy"]
        c2a = ds_results["C2_aligned"]["accuracy"]
        c2s = ds_results["C2_shuffled"]["accuracy"]
        gap = c1 - c2a
        delta_vs_shuffled = c2a - c2s

        ds_results["transfer_gap"] = gap
        ds_results["aligned_minus_shuffled"] = delta_vs_shuffled
        all_results["datasets"][ds_name] = ds_results

        print(f"\n  C1 (genuine):         {c1:.3f}")
        print(f"  C2 (aligned proj):    {c2a:.3f}  (gap={gap:+.3f})")
        print(f"  C2 (shuffled proj):   {c2s:.3f}")
        if c2a > c2s + 0.03:
            print(f"  → Aligned beats shuffled by {delta_vs_shuffled:.3f}. Projection captures real structure.")
        else:
            print(f"  → Aligned ≈ shuffled. Projection may not be meaningful.")

    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")

    del gemma_model
    torch.cuda.empty_cache()
    gc.collect()
    return all_results


if __name__ == "__main__":
    run_exp1b_pca_ridge()
