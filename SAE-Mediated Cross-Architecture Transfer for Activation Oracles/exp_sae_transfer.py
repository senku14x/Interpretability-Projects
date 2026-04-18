"""
Exp 0 & Exp 1
=============================================

Changes from v3:
  1. Llama loads with attn_implementation="sdpa" instead of flash_attention_2.
  2. Uses a local patched model loader inside this script; no repo file edits needed.
  3. SAE.from_pretrained() updated to the non-deprecated return style.
  4. Removed ignored generation arg "temperature" when do_sample=False.

Run full script:
  cd /content/activation_oracles
  python exp_sae_transfer.py

Resume only Exp 1 after Exp 0 already finished:
  import importlib.util
  spec = importlib.util.spec_from_file_location("exp_sae_transfer", "/content/activation_oracles/exp_sae_transfer.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  mod.run_exp1()
"""

import os
import gc
import json
import random
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch.nn as nn
from peft import LoraConfig
from transformers import AutoModelForCausalLM

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import (
    load_model as repo_load_model,
    load_tokenizer,
    layer_percent_to_layer,
)
from nl_probes.utils.eval import run_evaluation, score_eval_responses, proportion_confidence
from nl_probes.utils.dataset_utils import TrainingDataPoint

# =============================================================================
# CONFIG
# =============================================================================

GEMMA_MODEL = "google/gemma-2-9b-it"
GEMMA_ORACLE_ADAPTER = (
    "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_gemma-2-9b-it"
)
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# --- SAE ---
GEMMA_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
GEMMA_SAE_ID = "layer_21/width_16k/canonical"

# --- Eval settings ---
INJECTION_LAYER = 1
LAYER_PERCENT = 50
STEERING_COEFFICIENT = 1.0
DTYPE = torch.bfloat16
EVAL_BATCH_SIZE = 64
SEED = 42

GENERATION_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 10,
}

# --- Dataset config ---
NUM_TEST = 250
NUM_PROJ_TRAIN = 500

EVAL_DATASETS = {
    "sst2": {"num_test": NUM_TEST},
    "ag_news": {"num_test": NUM_TEST},
}

PROJ_TRAIN_DATASETS = {
    "sst2": {"num_train": NUM_PROJ_TRAIN},
    "ag_news": {"num_train": NUM_PROJ_TRAIN},
}

RESULTS_DIR = "exp_sae_transfer_results"
device = torch.device("cuda")

# =============================================================================
# PATCHED MODEL LOADER
# =============================================================================

def load_model_patched(model_name: str, dtype: torch.dtype):
    """
    Use the repo loader for Gemma, but override Llama to use SDPA instead of
    flash_attention_2 so it works in Colab without flash-attn.
    """
    lower = model_name.lower()
    if "llama" in lower:
        print("🧠 Loading model (patched: sdpa attention)...")
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
    return repo_load_model(model_name, dtype)

# =============================================================================
# HELPERS
# =============================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_cls_config(ds_name, num_train, num_test, splits, model_name):
    cls_config = ClassificationDatasetConfig(
        classification_dataset_name=ds_name,
        max_end_offset=-3,
        min_end_offset=-3,
        max_window_size=1,
        min_window_size=1,
    )
    ds_config = DatasetLoaderConfig(
        custom_dataset_params=cls_config,
        num_train=num_train,
        num_test=num_test,
        splits=splits,
        model_name=model_name,
        layer_percents=[LAYER_PERCENT],
        save_acts=True,
        batch_size=EVAL_BATCH_SIZE,
        seed=SEED,
    )
    return ds_config


def load_eval_data(model_name: str, datasets: dict, model=None) -> dict[str, list[TrainingDataPoint]]:
    out = {}
    for ds_name, cfg in datasets.items():
        ds_config = _make_cls_config(
            ds_name,
            num_train=0,
            num_test=cfg["num_test"],
            splits=["test"],
            model_name=model_name,
        )
        loader = ClassificationDatasetLoader(
            dataset_config=ds_config,
            model=model,
        )
        data = loader.load_dataset("test")
        assert data[0].steering_vectors is not None, f"No cached acts for {ds_name}"
        out[ds_name] = data
        print(f"  {ds_name} test: {len(data)} examples, sv shape={data[0].steering_vectors.shape}")
    return out


def load_proj_train_data(model_name: str, datasets: dict, model=None) -> dict[str, list[TrainingDataPoint]]:
    out = {}
    for ds_name, cfg in datasets.items():
        ds_config = _make_cls_config(
            ds_name,
            num_train=cfg["num_train"],
            num_test=0,
            splits=["train"],
            model_name=model_name,
        )
        loader = ClassificationDatasetLoader(
            dataset_config=ds_config,
            model=model,
        )
        data = loader.load_dataset("train")
        assert data[0].steering_vectors is not None, f"No cached acts for {ds_name} train"
        out[ds_name] = data
        print(f"  {ds_name} train: {len(data)} examples, sv shape={data[0].steering_vectors.shape}")
    return out


def assert_aligned(gemma_data: list[TrainingDataPoint], llama_data: list[TrainingDataPoint], label: str):
    assert len(gemma_data) == len(llama_data), (
        f"[{label}] Count mismatch: {len(gemma_data)} vs {len(llama_data)}"
    )
    mismatches = 0
    for i, (g, l) in enumerate(zip(gemma_data, llama_data)):
        if g.target_output != l.target_output:
            mismatches += 1
            if mismatches <= 3:
                print(
                    f"  [{label}] Label mismatch at idx {i}: "
                    f"gemma='{g.target_output}' vs llama='{l.target_output}'"
                )
    if mismatches > 0:
        raise ValueError(
            f"[{label}] {mismatches}/{len(gemma_data)} examples have mismatched labels. "
            "Gemma/Llama datasets are NOT aligned."
        )
    print(f"  [{label}] All {len(gemma_data)} examples aligned ✓")


def transform_steering_vectors(data: list[TrainingDataPoint], fn) -> list[TrainingDataPoint]:
    out = []
    for dp in data:
        dp_new = dp.model_copy(deep=True)
        if dp_new.steering_vectors is not None:
            dp_new.steering_vectors = fn(dp_new.steering_vectors)
        out.append(dp_new)
    return out


def run_oracle_eval(model, tokenizer, submodule, eval_data, lora_path: str | None, label: str) -> dict:
    print(f"\n  Running: {label} ({len(eval_data)} examples)...")
    results = run_evaluation(
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        device=device,
        dtype=DTYPE,
        global_step=-1,
        lora_path=lora_path,
        eval_batch_size=EVAL_BATCH_SIZE,
        steering_coefficient=STEERING_COEFFICIENT,
        generation_kwargs=GENERATION_KWARGS,
    )
    fmt_pct, acc_pct = score_eval_responses(results, eval_data)
    n = len(eval_data)
    correct = int(round(acc_pct * n))
    p, se, ci_lo, ci_hi = proportion_confidence(correct, n)
    print(f"    {label}: acc={p:.3f} [{ci_lo:.3f},{ci_hi:.3f}], fmt={fmt_pct:.3f}, n={n}")
    return {
        "label": label,
        "accuracy": p,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "format_correct": fmt_pct,
        "n": n,
    }


def fit_projection_f32(X: torch.Tensor, Y: torch.Tensor):
    X_f32 = X.float()
    Y_f32 = Y.float()
    ones = torch.ones(X_f32.shape[0], 1, device=X_f32.device)
    X_aug = torch.cat([X_f32, ones], dim=1)

    result = torch.linalg.lstsq(X_aug, Y_f32)
    W_aug = result.solution

    Y_pred = X_aug @ W_aug
    mse = ((Y_f32 - Y_pred) ** 2).mean().item()
    cosine = nn.functional.cosine_similarity(Y_f32, Y_pred, dim=-1).mean().item()

    return W_aug.to(DTYPE), {"train_mse": mse, "train_cosine": cosine}


def apply_projection(W_aug: torch.Tensor, vecs_KD: torch.Tensor) -> torch.Tensor:
    vecs = vecs_KD.to(W_aug.device, W_aug.dtype)
    return (vecs @ W_aug[:-1] + W_aug[-1].unsqueeze(0)).cpu()

# =============================================================================
# EXP 0
# =============================================================================

def run_exp0():
    print("\n" + "=" * 70)
    print("EXP 0: SAE Reconstruction Tax")
    print("=" * 70)
    set_all_seeds(SEED)

    print("\nLoading SAE...")
    from sae_lens import SAE
    sae = SAE.from_pretrained(
        release=GEMMA_SAE_RELEASE,
        sae_id=GEMMA_SAE_ID,
        device=str(device),
    )
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    gemma_layer = layer_percent_to_layer(GEMMA_MODEL, LAYER_PERCENT)
    sae_layer = int(GEMMA_SAE_ID.split("/")[0].split("_")[1])
    assert gemma_layer == sae_layer, (
        f"LAYER MISMATCH: dataset extracts layer {gemma_layer}, SAE is for layer {sae_layer}"
    )
    print(f"  Layer verified: dataset and SAE both use layer {gemma_layer} ✓")

    print("\nLoading Gemma-2-9B-IT...")
    model = load_model_patched(GEMMA_MODEL, DTYPE)
    tokenizer = load_tokenizer(GEMMA_MODEL)
    model.add_adapter(LoraConfig(), adapter_name="default")
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    print("\nLoading eval data...")
    eval_data_by_ds = load_eval_data(GEMMA_MODEL, EVAL_DATASETS, model=model)

    print("\nSAE reconstruction on eval activations:")
    for ds_name, data in eval_data_by_ds.items():
        vecs = torch.cat([dp.steering_vectors for dp in data], dim=0)
        vecs = vecs.to(sae.device, sae.dtype)
        with torch.no_grad():
            enc = sae.encode(vecs)
            dec = sae.decode(enc)
            cos = nn.functional.cosine_similarity(vecs, dec, dim=-1).mean().item()
            mse = ((vecs - dec) ** 2).mean().item()
            l0 = (enc > 0).float().sum(-1).mean().item()
        print(f"  {ds_name}: cosine={cos:.4f}, mse={mse:.4f}, L0={l0:.0f}")

    def sae_roundtrip(sv):
        sv = sv.to(sae.device, sae.dtype)
        with torch.no_grad():
            return sae.decode(sae.encode(sv)).cpu()

    def zero_steering(sv):
        return torch.zeros_like(sv)

    all_results = {}
    for ds_name, data in eval_data_by_ds.items():
        print(f"\n{'='*50}\nDataset: {ds_name}\n{'='*50}")
        ds_results = {}

        ds_results["C1_genuine"] = run_oracle_eval(
            model, tokenizer, submodule, data,
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"C1_genuine_{ds_name}",
        )

        ds_results["C0_sae"] = run_oracle_eval(
            model, tokenizer, submodule,
            transform_steering_vectors(data, sae_roundtrip),
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"C0_sae_{ds_name}",
        )

        ds_results["Ctrl_zero"] = run_oracle_eval(
            model, tokenizer, submodule,
            transform_steering_vectors(data, zero_steering),
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"Ctrl_zero_{ds_name}",
        )

        perm = torch.randperm(len(data))
        all_sv = [data[i].steering_vectors for i in range(len(data))]
        shuffled_sv = [all_sv[perm[i]] for i in range(len(data))]
        shuffled_data = []
        for i, dp in enumerate(data):
            dp_new = dp.model_copy(deep=True)
            dp_new.steering_vectors = shuffled_sv[i]
            shuffled_data.append(dp_new)
        ds_results["Ctrl_shuffle"] = run_oracle_eval(
            model, tokenizer, submodule, shuffled_data,
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"Ctrl_shuffle_{ds_name}",
        )

        c1 = ds_results["C1_genuine"]["accuracy"]
        c0 = ds_results["C0_sae"]["accuracy"]
        tax = c1 - c0
        tax_rel = tax / c1 if c1 > 0 else float("inf")
        ds_results["sae_tax_abs"] = tax
        ds_results["sae_tax_rel"] = tax_rel
        all_results[ds_name] = ds_results

        print(f"\n  C1 (genuine):  {c1:.3f}")
        print(f"  C0 (SAE):      {c0:.3f}  (tax={tax:+.3f}, {tax_rel:.0%})")
        print(f"  Ctrl_zero:     {ds_results['Ctrl_zero']['accuracy']:.3f}")
        print(f"  Ctrl_shuffle:  {ds_results['Ctrl_shuffle']['accuracy']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "exp0_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/exp0_results.json")

    del model, tokenizer, sae
    torch.cuda.empty_cache()
    gc.collect()
    return all_results

# =============================================================================
# EXP 1
# =============================================================================

def run_exp1():
    print("\n" + "=" * 70)
    print("EXP 1: Raw Cross-Architecture Projection")
    print("=" * 70)
    set_all_seeds(SEED)

    gemma_layer = layer_percent_to_layer(GEMMA_MODEL, LAYER_PERCENT)
    llama_layer = layer_percent_to_layer(LLAMA_MODEL, LAYER_PERCENT)
    print(f"Gemma layer: {gemma_layer}, Llama layer: {llama_layer}")

    print("\nPhase 1a: Gemma data...")
    gemma_model = load_model_patched(GEMMA_MODEL, DTYPE)
    gemma_eval = load_eval_data(GEMMA_MODEL, EVAL_DATASETS, model=gemma_model)
    gemma_proj_train = load_proj_train_data(GEMMA_MODEL, PROJ_TRAIN_DATASETS, model=gemma_model)

    llama_needs_extraction = False
    for ds_name in list(EVAL_DATASETS) + list(PROJ_TRAIN_DATASETS):
        num_train = PROJ_TRAIN_DATASETS.get(ds_name, {}).get("num_train", 0)
        num_test = EVAL_DATASETS.get(ds_name, {}).get("num_test", 0)
        for split, n in [("test", num_test), ("train", num_train)]:
            if n == 0:
                continue
            cfg = _make_cls_config(ds_name, num_train, num_test, [split], LLAMA_MODEL)
            loader = ClassificationDatasetLoader(dataset_config=cfg)
            path = os.path.join(loader.dataset_config.dataset_folder, loader.get_dataset_filename(split))
            if not os.path.exists(path):
                llama_needs_extraction = True
                break

    if llama_needs_extraction:
        print("\nPhase 1b: Llama caches missing. Freeing Gemma, extracting Llama acts...")
        del gemma_model
        torch.cuda.empty_cache()
        gc.collect()

        llama_model = load_model_patched(LLAMA_MODEL, DTYPE)
        load_eval_data(LLAMA_MODEL, EVAL_DATASETS, model=llama_model)
        load_proj_train_data(LLAMA_MODEL, PROJ_TRAIN_DATASETS, model=llama_model)
        del llama_model
        torch.cuda.empty_cache()
        gc.collect()

        print("\nReloading Gemma...")
        gemma_model = load_model_patched(GEMMA_MODEL, DTYPE)
    else:
        print("\nPhase 1b: All Llama caches found ✓")

    llama_eval = load_eval_data(LLAMA_MODEL, EVAL_DATASETS)
    llama_proj_train = load_proj_train_data(LLAMA_MODEL, PROJ_TRAIN_DATASETS)

    print("\nPhase 2: Asserting Gemma/Llama alignment...")
    for ds_name in EVAL_DATASETS:
        assert_aligned(gemma_eval[ds_name], llama_eval[ds_name], f"{ds_name}_test")
    for ds_name in PROJ_TRAIN_DATASETS:
        assert_aligned(gemma_proj_train[ds_name], llama_proj_train[ds_name], f"{ds_name}_train")

    print("\nPhase 3: Fitting projection on train split...")
    gemma_train_vecs = torch.cat(
        [torch.cat([dp.steering_vectors for dp in gemma_proj_train[ds]]) for ds in PROJ_TRAIN_DATASETS],
        dim=0,
    )
    llama_train_vecs = torch.cat(
        [torch.cat([dp.steering_vectors for dp in llama_proj_train[ds]]) for ds in PROJ_TRAIN_DATASETS],
        dim=0,
    )
    print(f"  Projection training: {gemma_train_vecs.shape[0]} vectors")
    print(f"  Gemma dim: {gemma_train_vecs.shape[1]}, Llama dim: {llama_train_vecs.shape[1]}")

    W_aligned, aligned_stats = fit_projection_f32(
        llama_train_vecs.to(device), gemma_train_vecs.to(device)
    )
    print(f"  Aligned projection: {aligned_stats}")

    perm = torch.randperm(gemma_train_vecs.shape[0])
    W_shuffled, shuffled_stats = fit_projection_f32(
        llama_train_vecs.to(device), gemma_train_vecs[perm].to(device)
    )
    print(f"  Shuffled projection: {shuffled_stats}")

    for ds_name in EVAL_DATASETS:
        g_vecs = torch.cat([dp.steering_vectors for dp in gemma_eval[ds_name]], dim=0).to(device)
        l_vecs = torch.cat([dp.steering_vectors for dp in llama_eval[ds_name]], dim=0).to(device)
        pred_aligned = apply_projection(W_aligned, l_vecs)
        pred_shuffled = apply_projection(W_shuffled, l_vecs)
        cos_a = nn.functional.cosine_similarity(g_vecs.cpu(), pred_aligned, dim=-1).mean().item()
        cos_s = nn.functional.cosine_similarity(g_vecs.cpu(), pred_shuffled, dim=-1).mean().item()
        print(f"  {ds_name} eval: aligned_cos={cos_a:.4f}, shuffled_cos={cos_s:.4f}")

    print("\nPhase 4: Oracle evaluation...")
    gemma_tokenizer = load_tokenizer(GEMMA_MODEL)
    gemma_model.add_adapter(LoraConfig(), adapter_name="default")
    submodule = get_hf_submodule(gemma_model, INJECTION_LAYER)

    all_results = {}
    for ds_name in EVAL_DATASETS:
        print(f"\n{'='*50}\nDataset: {ds_name}\n{'='*50}")
        g_data = gemma_eval[ds_name]
        l_data = llama_eval[ds_name]
        ds_results = {}

        ds_results["C1_genuine"] = run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, g_data,
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"C1_{ds_name}",
        )

        proj_data = []
        for g_dp, l_dp in zip(g_data, l_data):
            dp_new = g_dp.model_copy(deep=True)
            dp_new.steering_vectors = apply_projection(W_aligned, l_dp.steering_vectors)
            proj_data.append(dp_new)
        ds_results["C2_aligned"] = run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, proj_data,
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"C2_aligned_{ds_name}",
        )

        shuf_data = []
        for g_dp, l_dp in zip(g_data, l_data):
            dp_new = g_dp.model_copy(deep=True)
            dp_new.steering_vectors = apply_projection(W_shuffled, l_dp.steering_vectors)
            shuf_data.append(dp_new)
        ds_results["C2_shuffled"] = run_oracle_eval(
            gemma_model, gemma_tokenizer, submodule, shuf_data,
            lora_path=GEMMA_ORACLE_ADAPTER, label=f"C2_shuffled_{ds_name}",
        )

        c1 = ds_results["C1_genuine"]["accuracy"]
        c2a = ds_results["C2_aligned"]["accuracy"]
        c2s = ds_results["C2_shuffled"]["accuracy"]
        gap = c1 - c2a
        ds_results["transfer_gap"] = gap
        all_results[ds_name] = ds_results

        print(f"\n  C1 (genuine):         {c1:.3f}")
        print(f"  C2 (aligned proj):    {c2a:.3f}  (gap={gap:+.3f})")
        print(f"  C2 (shuffled proj):   {c2s:.3f}")
        if c2a > c2s + 0.03:
            print(f"  → Aligned beats shuffled by {c2a-c2s:.3f}. Projection captures real structure.")
        else:
            print(f"  → Aligned ≈ shuffled. Projection may not be meaningful.")

    with open(os.path.join(RESULTS_DIR, "exp1_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/exp1_results.json")

    del gemma_model
    torch.cuda.empty_cache()
    gc.collect()
    return all_results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_all_seeds(SEED)

    print("=" * 70)
    print("SAE-Mediated Cross-Architecture Transfer: Exp 0 & Exp 1 (v4)")
    print("=" * 70)
    print(f"Oracle:  {GEMMA_ORACLE_ADAPTER}")
    print(f"SAE:     {GEMMA_SAE_RELEASE} / {GEMMA_SAE_ID}")
    print(f"Layer:   {LAYER_PERCENT}%")
    print(f"Test:    {list(EVAL_DATASETS.keys())}, n={NUM_TEST} each")
    print(f"Proj:    {list(PROJ_TRAIN_DATASETS.keys())}, n={NUM_PROJ_TRAIN} each")

    exp0 = run_exp0()

    avg_c1 = sum(r["C1_genuine"]["accuracy"] for r in exp0.values()) / len(exp0)
    avg_tax = sum(r["sae_tax_rel"] for r in exp0.values()) / len(exp0)
    avg_zero = sum(r["Ctrl_zero"]["accuracy"] for r in exp0.values()) / len(exp0)

    print(f"\n\nEXP 0 GATE:")
    print(f"  avg C1={avg_c1:.3f}, avg C0_tax={avg_tax:.0%}, avg Ctrl_zero={avg_zero:.3f}")

    if avg_c1 < 0.55:
        print("  STOP: Oracle baseline too weak.")
    elif avg_c1 - avg_zero < 0.05:
        print("  STOP: Oracle barely beats zero-steering. Activations not used.")
    else:
        if avg_tax > 0.30:
            print(f"  CAUTION: High SAE tax ({avg_tax:.0%}). Proceeding anyway for signal.")
        else:
            print(f"  OK: SAE tax manageable ({avg_tax:.0%}).")
        exp1 = run_exp1()
