import functools
import sys
from pathlib import Path
from typing import Callable

import circuitsvis as cv
import einops
import numpy as np
import plotly.io as pio
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display

# Plotly remote viewer: writes each fig.show() to /tmp/plotly_html/latest.html
# served on port 8050. SSH with -L 8050:localhost:8050, then open localhost:8050
_plotly_dir = Path("/tmp/plotly_html")
_plotly_dir.mkdir(exist_ok=True)
_orig_fig_show = pio.show

def _remote_fig_show(fig, *args, **kwargs):
    path = _plotly_dir / "latest.html"
    fig.write_html(str(path), auto_open=False)
    print(f"Plot written to {path} — refresh browser at http://localhost:8050")

pio.show = _remote_fig_show
import plotly.graph_objects as _go
_go.Figure.show = lambda self, *a, **kw: _remote_fig_show(self, *a, **kw)
from jaxtyping import Float, Int
from torch import T_co, Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

import plotly.io as pio
pio.renderers.default = "png"

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"



cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)


def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)



A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dim: {AB_factor.rdim}, Left dim: {AB_factor.ldim}, Hidden dim: {AB_factor.mdim}")

print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)

print("\nSingular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)

print("\nFull SVD:")
print(AB_factor.svd())


C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C

print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
print(f"\nRight dim: {ABC_factor.rdim}, Left dim: {ABC_factor.ldim}, Hidden dim: {ABC_factor.mdim}")


head_index = 4
layer = 1

# W_OV = W_V W_O
# W_QK = W_Q (W_K^T)


W_OV_1_4 = FactoredMatrix(model.W_V[1, 4], model.W_O[1, 4])

full_OV_circuit = (model.W_E @ W_OV_1_4) @ model.W_U


tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)


indices = t.randint(0, model.cfg.d_vocab, (200,))
full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

imshow(
    full_OV_circuit_sample,
    labels={"x": "Logits on output token", "y": "Input token"},
    title="Full OV circuit for copying head",
    width=700,
    height=600,
)


def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 1000) -> float:
    """
    Return the fraction of the time that the maximum value is on the circuit diagonal.
    """
    total = 0
    for indices in t.split(t.arange(full_OV_circuit.shape[0], device=device), batch_size):
        AB_slice = full_OV_circuit[indices].AB
        total += (t.argmax(AB_slice, dim=1) == indices).float().sum().item()
    return total / full_OV_circuit.shape[0]


print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(full_OV_circuit):.4f}")



print(model.W_V[1,4].shape)
print(model.W_O[1,4].shape)


W_OV_all = FactoredMatrix(t.cat((model.W_V[1, 4], model.W_V[1, 10]), dim=1), t.cat((model.W_O[1, 4], model.W_O[1, 10]), dim=0))





complete_OV_circuit = (model.W_E @ (W_OV_all)) @ model.W_U

print(f"Fraction of time that the best logit is on diagonal: {top_1_acc(complete_OV_circuit):.4f}")




layer = 0
head_index = 7

# Compute full QK matrix (for positional embeddings)
W_pos = model.W_pos
W_QK = model.W_Q[layer, head_index] @ model.W_K[layer, head_index].T
pos_by_pos_scores = W_pos @ W_QK @ W_pos.T

# Mask, scale and softmax the scores
mask = t.tril(t.ones_like(pos_by_pos_scores)).bool()
pos_by_pos_pattern = t.where(mask, pos_by_pos_scores / model.cfg.d_head**0.5, -1.0e6).softmax(-1)

# Plot the results
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")
imshow(
    utils.to_numpy(pos_by_pos_pattern[:200, :200]),
    labels={"x": "Key", "y": "Query"},
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700,
    height=600,
)



def decompose_qk_input(cache: ActivationCache) -> Float[Tensor, "n_heads+2 posn d_model"]:
    """
    Retrieves all the input tensors to the first attention layer, and concatenates them along the
    0th dim.

    The [i, :, :]th element is y_i (from notation above). The sum of these tensors along the 0th
    dim should be the input to the first attention layer.
    """
    raise NotImplementedError()


def decompose_q(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of query vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values).
    """
    raise NotImplementedError()


def decompose_k(
    decomposed_qk_input: Float[Tensor, "n_heads+2 posn d_model"],
    ind_head_index: int,
    model: HookedTransformer,
) -> Float[Tensor, "n_heads+2 posn d_head"]:
    """
    Computes the tensor of key vectors for each decomposed QK input.

    The [i, :, :]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    """
    raise NotImplementedError()


# Recompute rep tokens/logits/cache, if we haven't already
seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch_size)
rep_cache.remove_batch_dim()

ind_head_index = 4

# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index, model)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index, model)
t.testing.assert_close(
    decomposed_qk_input.sum(0),
    rep_cache["resid_pre", 1] + rep_cache["pos_embed"],
    rtol=0.01,
    atol=1e-05,
)
t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)

# Second, we plot our results
component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])),
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}",
        y=component_labels,
        width=800,
        height=400,
    )




















