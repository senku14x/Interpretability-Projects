"""
src/extract.py

Residual stream activation extraction using TransformerLens hooks.
All functions extract the final-token position, which is where the model's
representation of the full context lives when probing prompt activations.
"""

import os
import numpy as np
import torch
from typing import Dict, List
from transformer_lens import HookedTransformer

from src.model import PROBE_LAYERS, NO_THINK_SUFFIX, get_act_name


def extract_activations(
    model: HookedTransformer,
    tokenizer,
    text: str,
    layers: List[int] = PROBE_LAYERS,
    device: str = "cuda",
    disable_thinking: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Run a single forward pass and return residual stream activations
    at the final token for each target layer.

    disable_thinking: append /no_think suffix to suppress Qwen3 thinking
    mode. Leave True during all probe experiments.

    Returns:
        dict mapping layer_idx -> numpy array of shape (d_model,)
    """
    if disable_thinking and not text.endswith(NO_THINK_SUFFIX):
        text = text + NO_THINK_SUFFIX

    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    activations = {}

    def make_hook(layer_idx: int):
        def hook_fn(value, hook):
            # value shape: (batch, seq_len, d_model)
            activations[layer_idx] = value[0, -1, :].detach().cpu().float().numpy()
            return value
        return hook_fn

    hooks = [(get_act_name(l), make_hook(l)) for l in layers]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=hooks)

    return activations


def extract_batch(
    model: HookedTransformer,
    tokenizer,
    texts: List[str],
    layers: List[int] = PROBE_LAYERS,
    device: str = "cuda",
    batch_size: int = 4,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Extract activations for a list of texts, processing in mini-batches.

    Returns:
        dict mapping layer_idx -> numpy array of shape (N, d_model)
    """
    all_acts = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if verbose and i % 10 == 0:
            print(f"  Extracting {i+1}/{len(texts)}")
        acts = extract_activations(model, tokenizer, text, layers=layers, device=device)
        for l in layers:
            all_acts[l].append(acts[l])

    return {l: np.stack(all_acts[l]) for l in layers}


def save_activations(acts: Dict[int, np.ndarray], save_dir: str, prefix: str) -> None:
    """
    Save activation arrays to disk.
    Files are named: {save_dir}/{prefix}_layer{N}.npy
    """
    os.makedirs(save_dir, exist_ok=True)
    for layer, arr in acts.items():
        path = os.path.join(save_dir, f"{prefix}_layer{layer}.npy")
        np.save(path, arr)
    print(f"Saved {len(acts)} layer files to {save_dir}/")


def load_activations(save_dir: str, prefix: str, layers: List[int] = PROBE_LAYERS) -> Dict[int, np.ndarray]:
    """Load activations saved by save_activations()."""
    acts = {}
    for layer in layers:
        path = os.path.join(save_dir, f"{prefix}_layer{layer}.npy")
        acts[layer] = np.load(path)
    return acts
