"""
src/model.py

Loads Qwen3-14B as a TransformerLens HookedTransformer.

Qwen3 notes:
  - Not gated on HuggingFace, no HF_TOKEN required.
  - Has a thinking mode that emits <think>...</think> tokens before
    responding. This MUST be disabled for probe experiments; otherwise
    the activation structure changes and final-token extraction breaks.
    Disable via the chat template with enable_thinking=False, or by
    appending /no_think to plain-text prompts.
  - Qwen3-14B has 40 layers, d_model 5120.
"""

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-14B"

# Layers probed across the 40-layer model (early, middle, late thirds)
PROBE_LAYERS = [8, 14, 20, 26, 32, 38]

D_MODEL = 5120

# Suffix that disables Qwen3 thinking mode in plain-text prompts.
# Without this the final-token activation reflects end-of-thinking,
# not end-of-prompt, which breaks probing.
NO_THINK_SUFFIX = " /no_think"


def load_model(
    model_name: str = MODEL_NAME,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    """
    Load Qwen3-14B via TransformerLens in bfloat16 on GPU.
    Returns (model, tokenizer).
    """
    print(f"Loading {model_name} ...")
    print(f"  dtype : {dtype}  device: {device}")

    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded. {n_params:.1f}B params | layers={model.cfg.n_layers} | d_model={model.cfg.d_model}")
    return model, tokenizer


def get_act_name(layer: int) -> str:
    """TransformerLens hook name for resid_post at a given layer."""
    return f"blocks.{layer}.hook_resid_post"
