"""
Exp 3: Extension experiments
=============================

Four experiments built on top of Exp 2 infrastructure:

  Ext A — Llama 32x SAE
          Same full SAE path but using llama_scope_lxr_32x (L0 ~100-150)
          instead of 8x (L0=35). Tests whether source SAE sparsity is the
          primary bottleneck.

  Ext B — C3b TopK sweep
          The no-Llama-encode condition was only tested at TopK=165.
          Sweep k in [50, 109, 129, 165, 192, 207] to find the optimum
          and assess sensitivity.

  Ext C — Cross-dataset generalization
          Retrain projection on SST2 train split only, evaluate on AG News.
          Tests whether C2 raw captures general cross-arch structure or
          just task-specific geometry.

  Ext D — KNN feature correlation baseline
          For each Llama feature, find the single highest-correlated Gemma
          feature across training examples (greedy 1-NN in feature
          activation space). Apply this sparse map, decode, evaluate.
          No learned projection — pure feature-to-feature matching.

NOTE: All SAE-related work is precomputed before oracle eval so SAEs
      and oracle model are never in memory simultaneously.

Requires:
  - /content/activation_oracles/exp_sae_transfer.py
  - /content/activation_oracles/exp2_sae_vs_raw.py
  - Cached activations from previous runs
"""

import os
import gc
import json
import torch
import torch.nn as nn
import importlib.util

# ---------------------------------------------------------------
# Load base modules
# ---------------------------------------------------------------
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

base = _load_module("/content/activation_oracles/exp_sae_transfer.py", "base")
exp2 = _load_module("/content/activation_oracles/exp2_sae_vs_raw.py",  "exp2")

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
LLAMA_SAE_RELEASE_32X = "llama_scope_lxr_32x"
LLAMA_SAE_ID_32X      = "l16r_32x"

GEMMA_SAE_RELEASE_16K = base.GEMMA_SAE_RELEASE
GEMMA_SAE_ID_16K      = base.GEMMA_SAE_ID

C3B_TOPK_SWEEP  = [50, 109, 129, 165, 192, 207]
FEAT_PCA_RANK   = 256
FEAT_RIDGE_ALPHA = 1.0
RAW_PCA_RANK    = 256
RAW_RIDGE_ALPHA = 1.0

RESULTS_FILE = os.path.join(base.RESULTS_DIR, "exp3_extensions_results.json")


# ---------------------------------------------------------------
# Helpers (thin wrappers around exp2 versions)
# ---------------------------------------------------------------
load_sae              = exp2.load_sae
encode_batched        = exp2.encode_batched
decode_batched        = exp2.decode_batched
enforce_sae_prior     = exp2.enforce_sae_prior
fit_pca_ridge         = exp2.fit_pca_ridge_projection
apply_proj            = exp2.apply_proj
cat_vectors           = exp2.cat_vectors
replace_vectors       = exp2.replace_vectors
to_list               = exp2.to_list
sparsity_stats        = exp2.sparsity_stats


# ---------------------------------------------------------------
# Load caches (no model extraction needed)
# ---------------------------------------------------------------
def load_caches():
    print("Loading cached activations...")
    gemma_eval       = base.load_eval_data(base.GEMMA_MODEL, base.EVAL_DATASETS)
    gemma_proj_train = base.load_proj_train_data(base.GEMMA_MODEL, base.PROJ_TRAIN_DATASETS)
    llama_eval       = base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS)
    llama_proj_train = base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS)
    return gemma_eval, gemma_proj_train, llama_eval, llama_proj_train


# ---------------------------------------------------------------
# KNN feature correlation map
# ---------------------------------------------------------------
@torch.no_grad()
def compute_knn_feature_map(
    l_train_feats: torch.Tensor,   # [N, d_llama]
    g_train_feats: torch.Tensor,   # [N, d_gemma]
    chunk_size: int = 512,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each Llama feature, find the single Gemma feature with the
    highest Pearson-like correlation across training examples.

    Returns:
        knn_map    [d_llama]  — Gemma feature index for each Llama feature
        knn_scores [d_llama]  — correlation score of the best match

    Note on interpretation: with L0=35 out of 32768, most columns in
    l_train_feats are all-zero. The correlation of two mostly-zero columns
    is dominated by shared zeros rather than co-activation signal. Treat
    KNN results cautiously for inactive Llama features.
    """
    L = l_train_feats.float().to(device)   # [N, d_llama]
    G = g_train_feats.float().to(device)   # [N, d_gemma]

    # Column-normalize (each feature's activation profile becomes unit vector)
    L_col_norm = L.norm(dim=0, keepdim=True).clamp_min(1e-6)   # [1, d_llama]
    G_col_norm = G.norm(dim=0, keepdim=True).clamp_min(1e-6)   # [1, d_gemma]
    L_n = L / L_col_norm   # [N, d_llama]
    G_n = G / G_col_norm   # [N, d_gemma]

    # Identify truly active Llama features (non-zero across training set)
    l_active_mask = (L > 0).any(dim=0)   # [d_llama]
    n_active = l_active_mask.sum().item()
    print(f"  Active Llama features (non-zero in any train example): {n_active} / {L.shape[1]}")

    d_llama = L.shape[1]
    knn_map    = torch.zeros(d_llama, dtype=torch.long,  device="cpu")
    knn_scores = torch.full( (d_llama,), -1.0,           device="cpu")

    LT = L_n.T   # [d_llama, N]

    for start in range(0, d_llama, chunk_size):
        end   = min(start + chunk_size, d_llama)
        chunk = LT[start:end].to(device)          # [chunk, N]
        corr  = chunk @ G_n                        # [chunk, d_gemma]
        best_j     = corr.argmax(dim=1).cpu()      # [chunk]
        best_score = corr.max(dim=1).values.cpu()  # [chunk]
        knn_map[start:end]    = best_j
        knn_scores[start:end] = best_score

        if start % (chunk_size * 20) == 0:
            print(f"    KNN map progress: {end}/{d_llama}")

    del L, G, L_n, G_n, LT
    torch.cuda.empty_cache()

    # Summary stats on active features only
    active_scores = knn_scores[l_active_mask.cpu()]
    print(f"  Active feature correlation stats: "
          f"mean={active_scores.mean():.4f}, "
          f"median={active_scores.median():.4f}, "
          f"min={active_scores.min():.4f}, "
          f"max={active_scores.max():.4f}")

    return knn_map, knn_scores


@torch.no_grad()
def apply_knn_mapping(
    llama_feats: torch.Tensor,   # [N, d_llama]
    knn_map:     torch.Tensor,   # [d_llama]
    d_gemma:     int,
    topk_k:      int,
) -> torch.Tensor:
    """
    Map active Llama features to Gemma features via the KNN map,
    accumulate activation values, then enforce TopK sparsity.
    Returns [N, d_gemma].
    """
    N = llama_feats.shape[0]
    gemma_feats = torch.zeros(N, d_gemma, dtype=llama_feats.dtype)

    for i in range(N):
        active_idx = (llama_feats[i] > 0).nonzero(as_tuple=True)[0]
        if len(active_idx) == 0:
            continue
        mapped_idx = knn_map[active_idx]            # which Gemma features
        vals       = llama_feats[i, active_idx]     # activation magnitudes
        gemma_feats[i].scatter_add_(0, mapped_idx, vals)

    return enforce_sae_prior(gemma_feats, topk_k)


# ---------------------------------------------------------------
# Ext A: Llama 32x SAE
# ---------------------------------------------------------------
def compute_ext_a(gemma_proj_train, llama_proj_train, llama_eval):
    print("\n" + "="*60)
    print("Ext A: Llama 32x SAE")
    print("="*60)

    g_train_hid = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)
    l_train_hid = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)

    # Encode with Llama 32x SAE
    print(f"\nLoading Llama 32x SAE ({LLAMA_SAE_RELEASE_32X} / {LLAMA_SAE_ID_32X})...")
    llama_sae_32x = load_sae(LLAMA_SAE_RELEASE_32X, LLAMA_SAE_ID_32X, str(base.device))

    l_train_feats_32x = encode_batched(llama_sae_32x, l_train_hid)
    l_eval_feats_32x  = {ds: encode_batched(llama_sae_32x, cat_vectors(llama_eval[ds]))
                         for ds in base.EVAL_DATASETS}

    print(f"  Llama 32x feature sparsity (train): {sparsity_stats(l_train_feats_32x)}")
    del llama_sae_32x; torch.cuda.empty_cache(); gc.collect()

    # Encode Gemma 16k features
    print("\nLoading Gemma 16k SAE...")
    gemma_sae_16k = load_sae(GEMMA_SAE_RELEASE_16K, GEMMA_SAE_ID_16K, str(base.device))

    g_train_feats_16k = encode_batched(gemma_sae_16k, g_train_hid)
    native_l0 = (g_train_feats_16k > 0).float().sum(-1).mean().item()
    print(f"  Native Gemma 16k L0: {native_l0:.1f}")

    # Fit projection Llama32x_feats -> Gemma16k_feats
    print("\nFitting projection (Llama 32x feat -> Gemma 16k feat)...")
    proj_a, stats_a = fit_pca_ridge(
        l_train_feats_32x.to(base.device),
        g_train_feats_16k.to(base.device),
        FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)
    perm = torch.randperm(g_train_feats_16k.shape[0])
    proj_s, stats_s = fit_pca_ridge(
        l_train_feats_32x.to(base.device),
        g_train_feats_16k[perm].to(base.device),
        FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)

    # Compute hidden vectors for all TopK values
    vectors = {ds: {} for ds in base.EVAL_DATASETS}
    geom    = {}
    topk_sweep = [50, 129, 165, 207]

    for ds in base.EVAL_DATASETS:
        l_fea = l_eval_feats_32x[ds].to(base.device)
        pf_a  = apply_proj(proj_a, l_fea)
        pf_s  = apply_proj(proj_s, l_fea)
        print(f"  {ds} 32x projected sparsity (aligned): {sparsity_stats(pf_a)}")

        for k in topk_sweep:
            vectors[ds][f"A_32x_topk{k}_aligned"]  = to_list(decode_batched(gemma_sae_16k, enforce_sae_prior(pf_a, k)))
            vectors[ds][f"A_32x_topk{k}_shuffled"] = to_list(decode_batched(gemma_sae_16k, enforce_sae_prior(pf_s, k)))

        geom[f"{ds}_32x"] = {
            "feat_aligned_cos": nn.functional.cosine_similarity(
                encode_batched(gemma_sae_16k, cat_vectors({ds: []}.__class__())).cpu()
                if False else  # placeholder — computed below
                g_train_feats_16k[:1].cpu(),
                pf_a[:1], dim=-1).mean().item(),
            "sparsity": sparsity_stats(pf_a),
            "native_l0": native_l0,
        }

    del gemma_sae_16k; torch.cuda.empty_cache(); gc.collect()

    return vectors, geom, {"aligned": stats_a, "shuffled": stats_s}


# ---------------------------------------------------------------
# Ext B: C3b TopK sweep
# ---------------------------------------------------------------
def compute_ext_b(gemma_proj_train, llama_proj_train, llama_eval):
    print("\n" + "="*60)
    print("Ext B: C3b (no Llama encode) TopK sweep")
    print("="*60)

    g_train_hid = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)
    l_train_hid = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)

    print("\nLoading Gemma 16k SAE...")
    gemma_sae_16k = load_sae(GEMMA_SAE_RELEASE_16K, GEMMA_SAE_ID_16K, str(base.device))

    g_train_feats = encode_batched(gemma_sae_16k, g_train_hid)
    native_l0     = (g_train_feats > 0).float().sum(-1).mean().item()
    print(f"  Native Gemma 16k L0: {native_l0:.1f}")

    # Fit projection: Llama hidden -> Gemma feature space (same as C3b in Exp 2)
    print("Fitting C3b projection (Llama hidden -> Gemma feat)...")
    proj_a, stats_a = fit_pca_ridge(
        l_train_hid.to(base.device),
        g_train_feats.to(base.device),
        FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)
    perm = torch.randperm(g_train_feats.shape[0])
    proj_s, stats_s = fit_pca_ridge(
        l_train_hid.to(base.device),
        g_train_feats[perm].to(base.device),
        FEAT_PCA_RANK, FEAT_RIDGE_ALPHA)

    vectors = {ds: {} for ds in base.EVAL_DATASETS}

    for ds in base.EVAL_DATASETS:
        l_hid = cat_vectors(llama_eval[ds]).to(base.device)
        pf_a  = apply_proj(proj_a, l_hid)
        pf_s  = apply_proj(proj_s, l_hid)

        for k in C3B_TOPK_SWEEP:
            vectors[ds][f"B_no_enc_topk{k}_aligned"]  = to_list(decode_batched(gemma_sae_16k, enforce_sae_prior(pf_a, k)))
            vectors[ds][f"B_no_enc_topk{k}_shuffled"] = to_list(decode_batched(gemma_sae_16k, enforce_sae_prior(pf_s, k)))

        print(f"  {ds}: decoded {len(C3B_TOPK_SWEEP)} TopK variants")

    del gemma_sae_16k; torch.cuda.empty_cache(); gc.collect()
    return vectors, {"aligned": stats_a, "shuffled": stats_s}


# ---------------------------------------------------------------
# Ext C: Cross-dataset generalization
# ---------------------------------------------------------------
def compute_ext_c(gemma_proj_train, llama_proj_train, gemma_eval, llama_eval):
    print("\n" + "="*60)
    print("Ext C: Cross-dataset generalization")
    print("  Train projection on SST2 only → test on AG News")
    print("  Train projection on AG News only → test on SST2")
    print("="*60)

    vectors = {ds: {} for ds in base.EVAL_DATASETS}
    geom    = {}

    for train_ds, test_ds in [("sst2", "ag_news"), ("ag_news", "sst2")]:
        print(f"\n  Train on: {train_ds}  |  Test on: {test_ds}")

        g_train = cat_vectors(gemma_proj_train[train_ds]).to(base.device)
        l_train = cat_vectors(llama_proj_train[train_ds]).to(base.device)
        g_test  = cat_vectors(gemma_eval[test_ds]).to(base.device)
        l_test  = cat_vectors(llama_eval[test_ds]).to(base.device)

        proj_a, stats_a = fit_pca_ridge(l_train, g_train, RAW_PCA_RANK, RAW_RIDGE_ALPHA)
        perm = torch.randperm(g_train.shape[0])
        proj_s, stats_s = fit_pca_ridge(l_train, g_train[perm], RAW_PCA_RANK, RAW_RIDGE_ALPHA)

        pred_a = apply_proj(proj_a, l_test, base.DTYPE)
        pred_s = apply_proj(proj_s, l_test, base.DTYPE)

        cos_a = nn.functional.cosine_similarity(g_test.cpu(), pred_a, dim=-1).mean().item()
        cos_s = nn.functional.cosine_similarity(g_test.cpu(), pred_s, dim=-1).mean().item()

        label = f"train{train_ds}_test{test_ds}"
        vectors[test_ds][f"C_xds_{label}_aligned"]  = to_list(pred_a)
        vectors[test_ds][f"C_xds_{label}_shuffled"] = to_list(pred_s)

        geom[label] = {
            "train_ds": train_ds, "test_ds": test_ds,
            "aligned_cos": cos_a, "shuffled_cos": cos_s,
            "train_stats": stats_a,
        }
        print(f"    aligned_cos={cos_a:.4f}, shuffled_cos={cos_s:.4f}")

    return vectors, geom


# ---------------------------------------------------------------
# Ext D: KNN feature correlation baseline
# ---------------------------------------------------------------
def compute_ext_d(gemma_proj_train, llama_proj_train, llama_eval):
    print("\n" + "="*60)
    print("Ext D: KNN feature correlation baseline")
    print("="*60)

    g_train_hid = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)
    l_train_hid = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0)

    # Encode with Llama 8x SAE
    print("\nLoading Llama 8x SAE...")
    llama_sae = load_sae("llama_scope_lxr_8x", "l16r_8x", str(base.device))
    # Direct load
    from sae_lens import SAE
    llama_sae = SAE.from_pretrained("llama_scope_lxr_8x", "l16r_8x", device=str(base.device))
    print(f"  Llama SAE: d_in={llama_sae.cfg.d_in} d_sae={llama_sae.cfg.d_sae}")

    l_train_feats = encode_batched(llama_sae, l_train_hid)
    l_eval_feats  = {ds: encode_batched(llama_sae, cat_vectors(llama_eval[ds]))
                     for ds in base.EVAL_DATASETS}
    del llama_sae; torch.cuda.empty_cache(); gc.collect()

    # Encode with Gemma 16k SAE
    print("\nLoading Gemma 16k SAE...")
    gemma_sae = load_sae(GEMMA_SAE_RELEASE_16K, GEMMA_SAE_ID_16K, str(base.device))
    d_gemma   = gemma_sae.cfg.d_sae

    g_train_feats = encode_batched(gemma_sae, g_train_hid)
    native_l0     = (g_train_feats > 0).float().sum(-1).mean().item()
    print(f"  Native Gemma 16k L0: {native_l0:.1f}")

    # Compute KNN map
    print("\nComputing KNN feature correlation map...")
    knn_map, knn_scores = compute_knn_feature_map(
        l_train_feats, g_train_feats, chunk_size=512, device=str(base.device))

    # Apply KNN map and decode for multiple TopK values
    vectors = {ds: {} for ds in base.EVAL_DATASETS}
    topk_sweep = [35, 50, 129, 165]   # 35 = Llama native L0

    for ds in base.EVAL_DATASETS:
        l_fea = l_eval_feats[ds].float()
        for k in topk_sweep:
            mapped = apply_knn_mapping(l_fea, knn_map, d_gemma, topk_k=k)
            decoded = decode_batched(gemma_sae, mapped)
            vectors[ds][f"D_knn_topk{k}_aligned"] = to_list(decoded)
            print(f"  {ds} KNN topk={k}: mapped sparsity={sparsity_stats(mapped)['eff_l0']:.1f}")

        # Also do shuffled KNN (random permutation of the map) as control
        perm_map = knn_map[torch.randperm(knn_map.shape[0])]
        for k in topk_sweep:
            mapped_s = apply_knn_mapping(l_fea, perm_map, d_gemma, topk_k=k)
            vectors[ds][f"D_knn_topk{k}_shuffled"] = to_list(decode_batched(gemma_sae, mapped_s))

    del gemma_sae; torch.cuda.empty_cache(); gc.collect()

    knn_stats = {
        "n_active_llama_features": int((l_train_feats > 0).any(dim=0).sum().item()),
        "mean_active_correlation": float(knn_scores[(l_train_feats > 0).any(dim=0)].mean().item()),
        "d_gemma": d_gemma,
    }
    return vectors, knn_stats


# ---------------------------------------------------------------
# Oracle eval
# ---------------------------------------------------------------
def run_oracle_all(gemma_eval, conditions_by_ds):
    """
    conditions_by_ds: ds -> {condition_name -> list of [1,D] tensors}
    Also adds C1 and C2 raw baselines for reference.
    """
    print("\n" + "="*60)
    print("Oracle evaluation")
    print("="*60)

    # Recompute raw baseline for reference
    gemma_proj_train = base.load_proj_train_data(base.GEMMA_MODEL, base.PROJ_TRAIN_DATASETS)
    llama_proj_train = base.load_proj_train_data(base.LLAMA_MODEL, base.PROJ_TRAIN_DATASETS)
    llama_eval_data  = base.load_eval_data(base.LLAMA_MODEL, base.EVAL_DATASETS)

    g_train = torch.cat([cat_vectors(gemma_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0).to(base.device)
    l_train = torch.cat([cat_vectors(llama_proj_train[ds]) for ds in base.PROJ_TRAIN_DATASETS], 0).to(base.device)
    proj_raw, _ = fit_pca_ridge(l_train, g_train, RAW_PCA_RANK, RAW_RIDGE_ALPHA)

    for ds in base.EVAL_DATASETS:
        l_vecs = cat_vectors(llama_eval_data[ds]).to(base.device)
        conditions_by_ds[ds]["REF_raw_aligned"] = to_list(apply_proj(proj_raw, l_vecs, base.DTYPE))

    # Load oracle
    gemma_model     = base.load_model_patched(base.GEMMA_MODEL, base.DTYPE)
    gemma_tokenizer = base.load_tokenizer(base.GEMMA_MODEL)
    gemma_model.add_adapter(base.LoraConfig(), adapter_name="default")
    submodule = base.get_hf_submodule(gemma_model, base.INJECTION_LAYER)

    results = {}
    for ds in base.EVAL_DATASETS:
        print(f"\n  Dataset: {ds}")
        g_data = gemma_eval[ds]
        ds_res = {}

        ds_res["C1_genuine"] = base.run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, g_data,
            lora_path=base.GEMMA_ORACLE_ADAPTER, label=f"C1_{ds}")

        c1 = ds_res["C1_genuine"]["accuracy"]

        for cname, vecs in conditions_by_ds[ds].items():
            ds_res[cname] = base.run_oracle_eval(
                gemma_model, gemma_tokenizer, submodule,
                replace_vectors(g_data, vecs),
                lora_path=base.GEMMA_ORACLE_ADAPTER,
                label=f"{cname}_{ds}")

        # Print summary
        def acc(k): return ds_res[k]["accuracy"] if k in ds_res else float("nan")
        c2 = acc("REF_raw_aligned")
        print(f"\n  C1 genuine:   {c1:.3f}")
        print(f"  C2 raw ref:   {c2:.3f}")

        # Ext A
        for k in [50, 129, 165, 207]:
            key = f"A_32x_topk{k}_aligned"
            if key in ds_res:
                ca, cs = acc(key), acc(f"A_32x_topk{k}_shuffled")
                rec = (ca - c2) / (c1 - c2) if abs(c1-c2) > 1e-8 else float("nan")
                print(f"  A 32x topk={k:<4}: {ca:.3f}  shuf={cs:.3f}  rec={rec:+.3f}")

        # Ext B
        for k in C3B_TOPK_SWEEP:
            key = f"B_no_enc_topk{k}_aligned"
            if key in ds_res:
                ca, cs = acc(key), acc(f"B_no_enc_topk{k}_shuffled")
                rec = (ca - c2) / (c1 - c2) if abs(c1-c2) > 1e-8 else float("nan")
                print(f"  B no-enc topk={k:<4}: {ca:.3f}  shuf={cs:.3f}  rec={rec:+.3f}")

        # Ext C
        for train_ds in base.EVAL_DATASETS:
            if train_ds == ds: continue
            key = f"C_xds_train{train_ds}_test{ds}_aligned"
            if key in ds_res:
                ca, cs = acc(key), acc(f"C_xds_train{train_ds}_test{ds}_shuffled")
                print(f"  C xds train={train_ds}: {ca:.3f}  shuf={cs:.3f}")

        # Ext D
        for k in [35, 50, 129, 165]:
            key = f"D_knn_topk{k}_aligned"
            if key in ds_res:
                ca, cs = acc(key), acc(f"D_knn_topk{k}_shuffled")
                print(f"  D knn topk={k:<4}: {ca:.3f}  shuf={cs:.3f}")

        results[ds] = ds_res

    del gemma_model; torch.cuda.empty_cache(); gc.collect()
    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_exp3():
    print("\n" + "="*60)
    print("Exp 3: Extension experiments")
    print("="*60)
    base.set_all_seeds(base.SEED)

    gemma_eval, gemma_proj_train, llama_eval, llama_proj_train = load_caches()

    # Collect all vectors per dataset
    all_vectors = {ds: {} for ds in base.EVAL_DATASETS}

    def merge(src):
        for ds in base.EVAL_DATASETS:
            if ds in src:
                all_vectors[ds].update(src[ds])

    # Ext A
    vecs_a, geom_a, stats_a = compute_ext_a(gemma_proj_train, llama_proj_train, llama_eval)
    merge(vecs_a)

    # Ext B
    vecs_b, stats_b = compute_ext_b(gemma_proj_train, llama_proj_train, llama_eval)
    merge(vecs_b)

    # Ext C
    vecs_c, geom_c = compute_ext_c(gemma_proj_train, llama_proj_train, gemma_eval, llama_eval)
    merge(vecs_c)

    # Ext D
    vecs_d, stats_d = compute_ext_d(gemma_proj_train, llama_proj_train, llama_eval)
    merge(vecs_d)

    # Oracle eval
    oracle_results = run_oracle_all(gemma_eval, all_vectors)

    out = {
        "config": {
            "llama_sae_32x": f"{LLAMA_SAE_RELEASE_32X}/{LLAMA_SAE_ID_32X}",
            "gemma_sae_16k": f"{GEMMA_SAE_RELEASE_16K}/{GEMMA_SAE_ID_16K}",
            "c3b_topk_sweep": C3B_TOPK_SWEEP,
        },
        "ext_a_geom":   geom_a,
        "ext_a_stats":  stats_a,
        "ext_c_geom":   geom_c,
        "ext_d_stats":  stats_d,
        "datasets":     oracle_results,
    }

    os.makedirs(base.RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")
    return out


if __name__ == "__main__":
    run_exp3()
