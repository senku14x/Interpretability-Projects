"""
Exp 2: SAE-mediated transfer vs strong raw baseline  (v2 — full ablation suite)
================================================================================

Conditions evaluated:
  C1               : genuine Gemma activations
  C2_raw           : Llama hidden -> PCA+ridge -> Gemma hidden  (strong raw baseline)
  C2_raw_shuffled  : same with permuted pairs (control)
  Ctrl_random      : norm-matched random Gaussian

  C3_dense         : full SAE path, dense decode (original broken version, kept for reference)
  C3_topk_{k}      : full SAE path, ReLU+TopK(k) before Gemma decode  [k in TOPK_VALUES]

  C3b_no_enc       : Llama hidden -> PCA+ridge -> Gemma FEATURE space -> TopK(165) -> Gemma decode
                     (ablates Llama SAE encode step)
  C3c_no_dec       : Llama SAE -> PCA+ridge -> Gemma HIDDEN space directly
                     (ablates Gemma SAE decode step)

  C3_131k_topk_{k} : full SAE path using 131k-wide Gemma SAE, TopK sweep

All SAE work is pre-computed and cached as hidden-space vectors before oracle eval,
so the oracle eval phase only needs the Gemma oracle model loaded (no SAEs in memory).

Notes:
  - Llama SAE (JumpReLU) native L0 ≈ 35; much sparser than Gemma's native L0 (129–207).
    The TopK sweep is calibrated to Gemma's native L0, but projected features derived from
    L0=35 source signals may not meaningfully support TopK=165+. Keep the sweep; note asymmetry.
  - 131k SAE uses average_l0_109 variant (closest to 16k SAE's native L0 of 192, medium
    sparsity range recommended by Gemma Scope). Makes width comparison clean: same approximate
    sparsity level, different dictionary size.
  - 131k SAEs are NOT in gemma-scope-9b-pt-res-canonical (16k only per layer).
    Use GEMMA_SAE_RELEASE_131K = "gemma-scope-9b-pt-res" for the 131k width.
"""

import os
import gc
import json
import torch
import torch.nn as nn
import importlib.util

# ---------------------------------------------------------------------
# Load base script as module
# ---------------------------------------------------------------------
BASE_PATH = "/content/activation_oracles/exp_sae_transfer.py"
spec = importlib.util.spec_from_file_location("exp_sae_transfer", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
RAW_PCA_RANK    = 256
RAW_RIDGE_ALPHA = 1.0
FEAT_PCA_RANK   = 256
FEAT_RIDGE_ALPHA = 1.0
ENCODE_BATCH    = 256
DECODE_BATCH    = 256

# TopK values to sweep for sparsity enforcement before Gemma SAE decode.
# Native Gemma 16k SAE L0 was 129-207 from Exp 0.
# NOTE: Llama SAE (JumpReLU) native L0 ≈ 35 — projected features may not
# meaningfully support the higher TopK values, but we keep the full sweep.
TOPK_VALUES = [50, 129, 165, 207]

GEMMA_SAE_RELEASE      = base.GEMMA_SAE_RELEASE          # gemma-scope-9b-pt-res-canonical
GEMMA_SAE_ID_16K       = base.GEMMA_SAE_ID               # layer_21/width_16k/canonical
GEMMA_SAE_RELEASE_131K = "gemma-scope-9b-pt-res"         # non-canonical release; has multiple widths incl. 131k
GEMMA_SAE_ID_131K      = "layer_21/width_131k/average_l0_109"

LLAMA_SAE_RELEASE   = "llama_scope_lxr_8x"
LLAMA_SAE_ID        = "l16r_8x"

RESULTS_FILE = os.path.join(base.RESULTS_DIR, "exp2_v2_full_ablation_results.json")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_sae(release, sae_id, device):
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    print(f"  Loaded SAE: {release} / {sae_id}  "
          f"d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae} arch={sae.cfg.architecture}")
    return sae


@torch.no_grad()
def encode_batched(sae, X, batch_size=256, out_dtype=torch.bfloat16):
    outs, dev = [], next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    for i in range(0, X.shape[0], batch_size):
        fb = sae.encode(X[i:i+batch_size].to(dev, sae_dtype))
        outs.append(fb.to(out_dtype).cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def decode_batched(sae, F, batch_size=256, out_dtype=torch.bfloat16):
    outs, dev = [], next(sae.parameters()).device
    sae_dtype = next(sae.parameters()).dtype
    for i in range(0, F.shape[0], batch_size):
        xb = sae.decode(F[i:i+batch_size].to(dev, sae_dtype))
        outs.append(xb.to(out_dtype).cpu())
    return torch.cat(outs, 0)


@torch.no_grad()
def enforce_sae_prior(feat: torch.Tensor, k: int) -> torch.Tensor:
    """ReLU then keep only top-k activations per vector."""
    feat = torch.relu(feat)
    if k <= 0:
        return feat
    topk_vals, topk_idx = feat.topk(k, dim=-1)
    sparse = torch.zeros_like(feat)
    sparse.scatter_(-1, topk_idx, topk_vals)
    return sparse


@torch.no_grad()
def fit_pca_ridge_projection(X, Y, rank, alpha):
    X, Y = X.float(), Y.float()
    mu_x  = X.mean(0, keepdim=True)
    std_x = X.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    mu_y  = Y.mean(0, keepdim=True)
    std_y = Y.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Xz, Yz = (X - mu_x) / std_x, (Y - mu_y) / std_y

    eff = min(rank, Xz.shape[0]-1, Xz.shape[1], Yz.shape[0]-1, Yz.shape[1])
    if eff < 2:
        raise ValueError(f"PCA rank too small: {eff}")

    _, _, Vx = torch.pca_lowrank(Xz, q=eff, center=False)
    _, _, Vy = torch.pca_lowrank(Yz, q=eff, center=False)
    Bx, By  = Vx[:, :eff], Vy[:, :eff]
    Xk, Yk  = Xz @ Bx, Yz @ By

    I = torch.eye(eff, device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(Xk.T @ Xk + alpha * I, Xk.T @ Yk)

    Yk_pred = Xk @ W
    Yz_pred = Yk_pred @ By.T
    Y_pred  = Yz_pred * std_y + mu_y

    stats = {
        "rank": eff, "alpha": alpha,
        "train_reduced_cosine": nn.functional.cosine_similarity(Yk, Yk_pred, dim=-1).mean().item(),
        "train_hidden_cosine":  nn.functional.cosine_similarity(Y,  Y_pred,  dim=-1).mean().item(),
        "train_hidden_mse":     ((Y - Y_pred)**2).mean().item(),
    }
    proj = dict(mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
                Bx=Bx, By=By, W=W, rank=eff, alpha=alpha)
    return proj, stats


@torch.no_grad()
def apply_proj(proj, X, out_dtype=torch.bfloat16):
    X  = X.to(proj["mu_x"].device).float()
    Xz = (X - proj["mu_x"]) / proj["std_x"]
    Xk = Xz @ proj["Bx"]
    Yk = Xk @ proj["W"]
    Yz = Yk @ proj["By"].T
    return (Yz * proj["std_y"] + proj["mu_y"]).to(out_dtype).cpu()


def cat_vectors(data_list):
    return torch.cat([dp.steering_vectors for dp in data_list], dim=0)


def replace_vectors(template_data, vecs_list):
    out = []
    for dp, sv in zip(template_data, vecs_list):
        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = sv
        out.append(dp_new)
    return out


def to_list(t: torch.Tensor):
    """Split [N, D] tensor into list of [1, D] tensors."""
    return [t[i:i+1] for i in range(t.shape[0])]


def sparsity_stats(feat: torch.Tensor) -> dict:
    return {
        "neg_frac":   (feat < 0).float().mean().item(),
        "eff_l0":     (feat > 0).float().sum(-1).mean().item(),
        "value_min":  feat.min().item(),
        "value_max":  feat.max().item(),
    }


# ---------------------------------------------------------------------
# Phase 1: ensure caches
# ---------------------------------------------------------------------

def ensure_caches():
    print("\nPhase 1a: Gemma data...")
    gemma_model = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    gemma_eval       = base.load_eval_data(base.GEMMA_MODEL, base.EVAL_DATASETS,      model=gemma_model)
    gemma_proj_train = base.load_proj_train_data(base.GEMMA_MODEL, base.PROJ_TRAIN_DATASETS, model=gemma_model)

    llama_needs = False
    for ds in list(base.EVAL_DATASETS) + list(base.PROJ_TRAIN_DATASETS):
        n_tr = base.PROJ_TRAIN_DATASETS.get(ds, {}).get("num_train", 0)
        n_te = base.EVAL_DATASETS.get(ds, {}).get("num_test", 0)
        for split, n in [("test", n_te), ("train", n_tr)]:
            if n == 0: continue
            cfg    = base._make_cls_config(ds, n_tr, n_te, [split], base.LLAMA_MODEL)
            loader = base.ClassificationDatasetLoader(dataset_config=cfg)
            path   = os.path.join(loader.dataset_config.dataset_folder,
                                  loader.get_dataset_filename(split))
            if not os.path.exists(path):
                llama_needs = True; break

    if llama_needs:
        print("\nPhase 1b: Llama caches missing — extracting...")
        del gemma_model; torch.cuda.empty_cache(); gc.collect()
        llama_model = base.load_model_patched(base.LLAMA_MODEL, base.DTYPE)
        base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS, model=llama_model)
        base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS, model=llama_model)
        del llama_model; torch.cuda.empty_cache(); gc.collect()
        print("\nReloading Gemma...")
        gemma_model = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    else:
        print("\nPhase 1b: All Llama caches found ✓")

    # reload from cache only (no model needed)
    del gemma_model; torch.cuda.empty_cache(); gc.collect()

    gemma_eval       = base.load_eval_data(base.GEMMA_MODEL, base.EVAL_DATASETS)
    gemma_proj_train = base.load_proj_train_data(base.GEMMA_MODEL, base.PROJ_TRAIN_DATASETS)
    llama_eval       = base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS)
    llama_proj_train = base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS)

    print("\nPhase 2: Asserting alignment...")
    for ds in base.EVAL_DATASETS:
        base.assert_aligned(gemma_eval[ds], llama_eval[ds], f"{ds}_test")
    for ds in base.PROJ_TRAIN_DATASETS:
        base.assert_aligned(gemma_proj_train[ds], llama_proj_train[ds], f"{ds}_train")

    return gemma_eval, gemma_proj_train, llama_eval, llama_proj_train


# ---------------------------------------------------------------------
# Phase 3a: raw baseline vectors
# ---------------------------------------------------------------------

def compute_raw_baseline(gemma_eval, gemma_proj_train, llama_eval, llama_proj_train):
    print("\nPhase 3a: Raw PCA+ridge baseline...")

    g_train = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0).to(base.device)
    l_train = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0).to(base.device)

    proj_a, stats_a = fit_pca_ridge_projection(l_train, g_train, RAW_PCA_RANK, RAW_RIDGE_ALPHA)
    perm = torch.randperm(g_train.shape[0])
    proj_s, stats_s = fit_pca_ridge_projection(l_train, g_train[perm], RAW_PCA_RANK, RAW_RIDGE_ALPHA)

    vectors = {}   # ds -> {condition -> list of [1,D] tensors}
    geom    = {}

    for ds in base.EVAL_DATASETS:
        g_vecs = cat_vectors(gemma_eval[ds]).to(base.device)
        l_vecs = cat_vectors(llama_eval[ds]).to(base.device)

        pred_a = apply_proj(proj_a, l_vecs, base.DTYPE)
        pred_s = apply_proj(proj_s, l_vecs, base.DTYPE)
        rand   = [v / v.norm().clamp_min(1e-6) * dp.steering_vectors.norm().clamp_min(1e-6)
                  for dp, v in zip(gemma_eval[ds], [torch.randn_like(dp.steering_vectors)
                                                    for dp in gemma_eval[ds]])]

        geom[ds] = {
            "aligned_cos":   nn.functional.cosine_similarity(g_vecs.cpu(), pred_a, dim=-1).mean().item(),
            "shuffled_cos":  nn.functional.cosine_similarity(g_vecs.cpu(), pred_s, dim=-1).mean().item(),
        }
        vectors[ds] = {
            "C2_raw_aligned":  to_list(pred_a),
            "C2_raw_shuffled": to_list(pred_s),
            "Ctrl_random":     rand,
        }
        print(f"  RAW {ds}: aligned_cos={geom[ds]['aligned_cos']:.4f}, "
              f"shuffled_cos={geom[ds]['shuffled_cos']:.4f}")

    return vectors, geom, {"aligned": stats_a, "shuffled": stats_s}


# ---------------------------------------------------------------------
# Phase 3b: all SAE conditions (16k and 131k)
# ---------------------------------------------------------------------

def compute_sae_conditions(gemma_eval, gemma_proj_train, llama_eval, llama_proj_train):
    """
    Returns:
        vectors: ds -> {condition_name -> list of [1,D] hidden tensors}
        geom:    diagnostics dict
        stats:   projection stats dict
    """
    print("\nPhase 3b: SAE-mediated conditions...")

    # ---- collect train/eval hidden vecs ----
    g_train_hid = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)
    l_train_hid = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)
    g_eval_hid  = {ds: cat_vectors(gemma_eval[ds]) for ds in base.EVAL_DATASETS}
    l_eval_hid  = {ds: cat_vectors(llama_eval[ds]) for ds in base.EVAL_DATASETS}

    # ================================================================
    # Llama SAE encode  (load, encode, unload)
    # ================================================================
    print("\n  Loading Llama SAE...")
    llama_sae = load_sae(LLAMA_SAE_RELEASE, LLAMA_SAE_ID, str(base.device))

    l_train_feats = encode_batched(llama_sae, l_train_hid)
    l_eval_feats  = {ds: encode_batched(llama_sae, l_eval_hid[ds]) for ds in base.EVAL_DATASETS}

    print(f"  Llama feature sparsity (train): {sparsity_stats(l_train_feats)}")

    del llama_sae; torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # Gemma 16k SAE  (encode train+eval features, fit proj, decode all)
    # ================================================================
    print("\n  Loading Gemma 16k SAE...")
    gemma_sae_16k = load_sae(GEMMA_SAE_RELEASE, GEMMA_SAE_ID_16K, str(base.device))

    g_train_feats_16k = encode_batched(gemma_sae_16k, g_train_hid)
    g_eval_feats_16k  = {ds: encode_batched(gemma_sae_16k, g_eval_hid[ds]) for ds in base.EVAL_DATASETS}

    native_l0_16k = (g_train_feats_16k > 0).float().sum(-1).mean().item()
    print(f"  Native Gemma 16k L0 (train): {native_l0_16k:.1f}")

    # ---- full SAE path projection (Llama feat -> Gemma 16k feat) ----
    print("\n  Fitting full SAE path projection (16k)...")
    proj_16k_a, stats_16k_a = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_feats_16k.to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)
    perm = torch.randperm(g_train_feats_16k.shape[0])
    proj_16k_s, stats_16k_s = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_feats_16k[perm].to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)

    # ---- no-Llama-encode path projection (Llama hidden -> Gemma 16k feat) ----
    print("  Fitting no-Llama-encode projection...")
    proj_noenc_a, stats_noenc_a = fit_pca_ridge_projection(
        l_train_hid.to(base.device), g_train_feats_16k.to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)
    proj_noenc_s, stats_noenc_s = fit_pca_ridge_projection(
        l_train_hid.to(base.device), g_train_feats_16k[perm].to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)

    # ---- decode all 16k conditions per eval dataset ----
    vectors = {ds: {} for ds in base.EVAL_DATASETS}
    geom    = {}

    for ds in base.EVAL_DATASETS:
        l_fea = l_eval_feats[ds].to(base.device)
        g_fea = g_eval_feats_16k[ds].to(base.device)
        l_hid_ev = l_eval_hid[ds].to(base.device)

        # projected features (dense, before any sparsity fix)
        pf_a = apply_proj(proj_16k_a, l_fea)     # [N, 16384] bfloat16 on cpu
        pf_s = apply_proj(proj_16k_s, l_fea)

        # no-enc projected features
        pf_noenc_a = apply_proj(proj_noenc_a, l_hid_ev)
        pf_noenc_s = apply_proj(proj_noenc_s, l_hid_ev)

        sparse_stats_a = sparsity_stats(pf_a)
        print(f"  {ds} projected 16k feat sparsity (aligned): {sparse_stats_a}")

        # decode dense (original broken condition — kept for reference)
        hid_dense_a = decode_batched(gemma_sae_16k, pf_a)
        hid_dense_s = decode_batched(gemma_sae_16k, pf_s)
        vectors[ds]["C3_dense_aligned"]  = to_list(hid_dense_a)
        vectors[ds]["C3_dense_shuffled"] = to_list(hid_dense_s)

        # TopK sweep
        for k in TOPK_VALUES:
            spa = enforce_sae_prior(pf_a, k)
            sps = enforce_sae_prior(pf_s, k)
            vectors[ds][f"C3_topk{k}_aligned"]  = to_list(decode_batched(gemma_sae_16k, spa))
            vectors[ds][f"C3_topk{k}_shuffled"] = to_list(decode_batched(gemma_sae_16k, sps))

        # no-Llama-encode, TopK=165
        spa_noenc_a = enforce_sae_prior(pf_noenc_a, 165)
        spa_noenc_s = enforce_sae_prior(pf_noenc_s, 165)
        vectors[ds]["C3b_no_enc_aligned"]  = to_list(decode_batched(gemma_sae_16k, spa_noenc_a))
        vectors[ds]["C3b_no_enc_shuffled"] = to_list(decode_batched(gemma_sae_16k, spa_noenc_s))

        # geometry
        geom[f"{ds}_16k"] = {
            "feat_aligned_cos":  nn.functional.cosine_similarity(g_fea.cpu(), pf_a,    dim=-1).mean().item(),
            "feat_shuffled_cos": nn.functional.cosine_similarity(g_fea.cpu(), pf_s,    dim=-1).mean().item(),
            "hid_dense_aligned_cos":  nn.functional.cosine_similarity(g_eval_hid[ds], hid_dense_a, dim=-1).mean().item(),
            "hid_dense_shuffled_cos": nn.functional.cosine_similarity(g_eval_hid[ds], hid_dense_s, dim=-1).mean().item(),
            "projected_feat_sparsity": sparse_stats_a,
            "native_l0_16k": native_l0_16k,
        }
        for k in TOPK_VALUES:
            hv = decode_batched(gemma_sae_16k, enforce_sae_prior(pf_a, k))
            geom[f"{ds}_16k"][f"hid_topk{k}_aligned_cos"] = nn.functional.cosine_similarity(
                g_eval_hid[ds], hv, dim=-1).mean().item()

    del gemma_sae_16k; torch.cuda.empty_cache(); gc.collect()

    # ================================================================
    # No-Gemma-decode path  (Llama SAE feat -> Gemma hidden directly)
    # ================================================================
    print("\n  Fitting no-Gemma-decode projection (Llama feat -> Gemma hidden)...")
    proj_nodec_a, stats_nodec_a = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_hid.to(base.device), RAW_PCA_RANK, RAW_RIDGE_ALPHA)
    proj_nodec_s, stats_nodec_s = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_hid[perm].to(base.device), RAW_PCA_RANK, RAW_RIDGE_ALPHA)

    for ds in base.EVAL_DATASETS:
        l_fea = l_eval_feats[ds].to(base.device)
        vectors[ds]["C3c_no_dec_aligned"]  = to_list(apply_proj(proj_nodec_a, l_fea))
        vectors[ds]["C3c_no_dec_shuffled"] = to_list(apply_proj(proj_nodec_s, l_fea))
        hid_a = apply_proj(proj_nodec_a, l_fea)
        geom[f"{ds}_nodec"] = {
            "hid_aligned_cos": nn.functional.cosine_similarity(g_eval_hid[ds], hid_a, dim=-1).mean().item(),
        }

    # ================================================================
    # Gemma 131k SAE  — uses GEMMA_SAE_RELEASE_131K, not the canonical release
    # average_l0_109 variant: closest to 16k SAE native L0 of 192, medium sparsity
    # ================================================================
    print("\n  Loading Gemma 131k SAE...")
    gemma_sae_131k = load_sae(GEMMA_SAE_RELEASE_131K, GEMMA_SAE_ID_131K, str(base.device))

    g_train_feats_131k = encode_batched(gemma_sae_131k, g_train_hid)
    g_eval_feats_131k  = {ds: encode_batched(gemma_sae_131k, g_eval_hid[ds]) for ds in base.EVAL_DATASETS}

    native_l0_131k = (g_train_feats_131k > 0).float().sum(-1).mean().item()
    k_131k = max(50, int(round(native_l0_131k)))
    print(f"  Native Gemma 131k L0 (train): {native_l0_131k:.1f}  → will use TopK={k_131k}")

    print("  Fitting full SAE path projection (131k)...")
    proj_131k_a, stats_131k_a = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_feats_131k.to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)
    perm131 = torch.randperm(g_train_feats_131k.shape[0])
    proj_131k_s, stats_131k_s = fit_pca_ridge_projection(
        l_train_feats.to(base.device), g_train_feats_131k[perm131].to(base.device), FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)

    for ds in base.EVAL_DATASETS:
        l_fea = l_eval_feats[ds].to(base.device)
        g_fea_131 = g_eval_feats_131k[ds].to(base.device)

        pf_a_131 = apply_proj(proj_131k_a, l_fea)
        pf_s_131 = apply_proj(proj_131k_s, l_fea)

        # sweep same TOPK_VALUES + native k for 131k
        topk_131k_sweep = sorted(set(TOPK_VALUES + [k_131k]))
        for k in topk_131k_sweep:
            spa = enforce_sae_prior(pf_a_131, k)
            sps = enforce_sae_prior(pf_s_131, k)
            vectors[ds][f"C3_131k_topk{k}_aligned"]  = to_list(decode_batched(gemma_sae_131k, spa))
            vectors[ds][f"C3_131k_topk{k}_shuffled"] = to_list(decode_batched(gemma_sae_131k, sps))

        geom[f"{ds}_131k"] = {
            "feat_aligned_cos":  nn.functional.cosine_similarity(g_fea_131.cpu(), pf_a_131, dim=-1).mean().item(),
            "feat_shuffled_cos": nn.functional.cosine_similarity(g_fea_131.cpu(), pf_s_131, dim=-1).mean().item(),
            "native_l0_131k": native_l0_131k,
            "native_topk_used": k_131k,
        }
        print(f"  {ds} 131k: feat_cos_aligned={geom[f'{ds}_131k']['feat_aligned_cos']:.4f}")

    del gemma_sae_131k; torch.cuda.empty_cache(); gc.collect()

    all_stats = {
        "full_sae_16k":  {"aligned": stats_16k_a,  "shuffled": stats_16k_s},
        "no_enc_16k":    {"aligned": stats_noenc_a, "shuffled": stats_noenc_s},
        "no_dec":        {"aligned": stats_nodec_a, "shuffled": stats_nodec_s},
        "full_sae_131k": {"aligned": stats_131k_a,  "shuffled": stats_131k_s},
    }

    return vectors, geom, all_stats


# ---------------------------------------------------------------------
# Phase 4: Oracle eval
# ---------------------------------------------------------------------

def run_oracle_eval_all(gemma_eval, raw_vectors, sae_vectors):
    """
    Runs oracle eval for every condition.
    raw_vectors:  ds -> {condition -> list of [1,D]}
    sae_vectors:  ds -> {condition -> list of [1,D]}
    """
    print("\nPhase 4: Oracle evaluation...")
    gemma_model    = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    gemma_tokenizer = base.load_tokenizer(base.GEMMA_MODEL)
    gemma_model.add_adapter(base.LoraConfig(), adapter_name="default")
    submodule = base.get_hf_submodule(gemma_model, base.INJECTION_LAYER)

    results = {}

    for ds in base.EVAL_DATASETS:
        print(f"\n{'='*55}\nDataset: {ds}\n{'='*55}")
        g_data    = gemma_eval[ds]
        ds_res    = {}
        all_vecs  = {**raw_vectors[ds], **sae_vectors[ds]}

        # C1 baseline (genuine Gemma activations)
        ds_res["C1_genuine"] = base.run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, g_data,
            lora_path=base.GEMMA_ORACLE_ADAPTER, label=f"C1_{ds}")

        c1 = ds_res["C1_genuine"]["accuracy"]

        for cond_name, vecs in all_vecs.items():
            ds_res[cond_name] = base.run_oracle_eval(
                gemma_model, gemma_tokenizer, submodule,
                replace_vectors(g_data, vecs),
                lora_path=base.GEMMA_ORACLE_ADAPTER,
                label=f"{cond_name}_{ds}")

        # --- print summary for key conditions ---
        def acc(k): return ds_res[k]["accuracy"] if k in ds_res else float("nan")

        c2  = acc("C2_raw_aligned")
        print(f"\n  C1  genuine:                {c1:.3f}")
        print(f"  C2  raw aligned:            {c2:.3f}")
        print(f"  C2  raw shuffled:           {acc('C2_raw_shuffled'):.3f}")
        print(f"  Ctrl random:                {acc('Ctrl_random'):.3f}")
        print(f"  C3  dense aligned:          {acc('C3_dense_aligned'):.3f}   [broken decode, reference]")
        for k in TOPK_VALUES:
            ca = acc(f"C3_topk{k}_aligned")
            cs = acc(f"C3_topk{k}_shuffled")
            rec = (ca - c2) / (c1 - c2) if abs(c1 - c2) > 1e-8 else float("nan")
            print(f"  C3  topk={k:<4} aligned:     {ca:.3f}  shuffled: {cs:.3f}  recovery: {rec:+.3f}")
        print(f"  C3b no-enc aligned:         {acc('C3b_no_enc_aligned'):.3f}  shuffled: {acc('C3b_no_enc_shuffled'):.3f}")
        print(f"  C3c no-dec aligned:         {acc('C3c_no_dec_aligned'):.3f}  shuffled: {acc('C3c_no_dec_shuffled'):.3f}")

        # best 131k condition
        for k in TOPK_VALUES:
            key = f"C3_131k_topk{k}_aligned"
            if key in ds_res:
                print(f"  C3  131k topk={k:<4} aligned: {acc(key):.3f}  shuffled: {acc(f'C3_131k_topk{k}_shuffled'):.3f}")

        results[ds] = ds_res

    del gemma_model; torch.cuda.empty_cache(); gc.collect()
    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_exp2_full():
    print("\n" + "="*70)
    print("EXP 2 v2: Full ablation suite")
    print("="*70)
    base.set_all_seeds(base.SEED)

    gemma_eval, gemma_proj_train, llama_eval, llama_proj_train = ensure_caches()

    raw_vectors, raw_geom, raw_stats = compute_raw_baseline(
        gemma_eval, gemma_proj_train, llama_eval, llama_proj_train)

    sae_vectors, sae_geom, sae_stats = compute_sae_conditions(
        gemma_eval, gemma_proj_train, llama_eval, llama_proj_train)

    oracle_results = run_oracle_eval_all(gemma_eval, raw_vectors, sae_vectors)

    out = {
        "config": {
            "raw_pca_rank": RAW_PCA_RANK, "raw_ridge_alpha": RAW_RIDGE_ALPHA,
            "feat_pca_rank": FEAT_PCA_RANK, "feat_ridge_alpha": FEAT_RIDGE_ALPHA,
            "topk_values": TOPK_VALUES,
            "llama_sae": f"{LLAMA_SAE_RELEASE}/{LLAMA_SAE_ID}",
            "gemma_sae_16k":  f"{GEMMA_SAE_RELEASE}/{GEMMA_SAE_ID_16K}",
            "gemma_sae_131k": f"{GEMMA_SAE_RELEASE_131K}/{GEMMA_SAE_ID_131K}",
        },
        "geometry": {**raw_geom, **sae_geom},
        "projection_stats": {**raw_stats, **sae_stats},
        "datasets": oracle_results,
    }

    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")
    return out


if __name__ == "__main__":
    run_exp2_full()
