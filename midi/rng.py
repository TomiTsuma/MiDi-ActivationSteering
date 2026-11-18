"""Keyed, counter-based randomness for reproducible per-molecule sampling.

Why this exists
---------------
The released MiDi sampler draws every random quantity from the shared global
torch stream:

  * ``DistributionNodes.sample_n``                -> the atom count
  * ``NoiseModel.sample_limit_dist``              -> the initial state
  * ``NoiseModel.sample_zs_from_zt_and_pred``     -> position noise, every step
  * ``diffusion_utils.sample_discrete_features``  -> categorical draws, every step

With a shared stream, molecule *i*'s randomness depends on how many draws
preceded it, so it changes with batch size, batch position and execution
order. Any later comparison between a steered and an unsteered arm is then
unpaired, silently.

This module makes every draw a pure function of ``(run_seed, molecule_index)``.
Molecule *i* receives identical randomness in every arm and configuration, no
matter how it was batched.

What this does NOT change
-------------------------
The distribution being sampled is unchanged. Gaussian noise is still standard
normal. Categorical draws are made by inverse-CDF lookup on a uniform variate,
which is distributionally identical to ``torch.multinomial(p, 1)`` but a pure
function of that variate, so it does not depend on any device RNG
implementation. ``tests/test_determinism.py`` checks the equivalence
empirically.

Notation used below
-------------------
    B      batch size (molecules sampled together)
    T      number of diffusion steps (500 for this checkpoint)
    n_max  padded atom count, held FIXED across batches (29 for QM9 with H)
    K      number of categories in a categorical draw
"""

import hashlib

import torch

_MASK63 = (1 << 63) - 1

# Purpose tags. Each names one independent stream per molecule, so adding a
# draw site later cannot desynchronise an existing one.
P_N_NODES = "n_nodes"
P_INIT_POS = "init_pos"
P_INIT_CAT = "init_cat"
P_STEP_POS = "step_pos"
P_STEP_CAT = "step_cat"


def derive_seed(run_seed: int, mol_id: int, purpose: str) -> int:
    """Counter-based seed: a pure function of its key.

    Uses blake2b from the standard library rather than Python's ``hash()``,
    which is randomised per process unless PYTHONHASHSEED is set and would
    therefore give different seeds on different runs.
    """
    key = f"{int(run_seed)}|{int(mol_id)}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & _MASK63


def _gen(run_seed: int, mol_id: int, purpose: str) -> torch.Generator:
    g = torch.Generator()  # CPU generator: deterministic across devices
    g.manual_seed(derive_seed(run_seed, mol_id, purpose))
    return g


# ----------------------------------------------------------------------------
# Categorical sampling by inverse CDF
# ----------------------------------------------------------------------------

def categorical_from_uniform(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Draw one category per row by inverse-CDF lookup.

    probs : (..., K) non-negative; each row is normalised internally
    u     : (...)    uniform in [0, 1)
    returns (..., 1) long, the same shape ``torch.multinomial(probs, 1)`` gives.

    For a row with CDF c, the result is the smallest index i with u < c[i].
    Categories of probability zero are never selected, because they leave the
    CDF unchanged and so span an empty interval.
    """
    cdf = probs.cumsum(dim=-1)
    last = cdf[..., -1:].clamp_min(torch.finfo(cdf.dtype).tiny)
    cdf = (cdf / last).contiguous()
    idx = torch.searchsorted(cdf, u.unsqueeze(-1).contiguous(), right=False)
    return idx.clamp_(max=probs.shape[-1] - 1)


def n_unif_per_step(n_max: int) -> int:
    """Uniforms consumed by one reverse step: X, charges, then the dense E."""
    return 2 * n_max + n_max * n_max


# ----------------------------------------------------------------------------
# Per-molecule atom count
# ----------------------------------------------------------------------------

def sample_n_nodes(run_seed: int, mol_ids, node_prob: torch.Tensor, device=None):
    """Per-molecule atom count, replacing ``DistributionNodes.sample_n``.

    node_prob : the histogram-derived probability vector, ``node_dist.prob``.
    Returns a long tensor of shape (len(mol_ids),).

    Uses inverse CDF rather than ``torch.multinomial`` so the result is stable
    across torch versions as well as across runs.
    """
    probs = node_prob.detach().to("cpu", torch.float64)
    cdf = probs.cumsum(0)
    cdf = cdf / cdf[-1].clamp_min(torch.finfo(cdf.dtype).tiny)
    out = torch.empty(len(mol_ids), dtype=torch.long)
    for i, mid in enumerate(mol_ids):
        u = torch.rand(1, generator=_gen(run_seed, mid, P_N_NODES), dtype=torch.float64)
        out[i] = int(torch.searchsorted(cdf, u).clamp_(max=probs.numel() - 1).item())
    return out if device is None else out.to(device)


# ----------------------------------------------------------------------------
# The noise tape
# ----------------------------------------------------------------------------

def tape_bytes(batch_size: int, T: int, n_max: int, itemsize: int = 4) -> int:
    """Bytes a tape occupies, for sizing the batch against GPU memory."""
    per_step = n_max * 3 + n_unif_per_step(n_max)
    return (T + 1) * batch_size * per_step * itemsize


class NoiseTape:
    """Pre-generated per-molecule randomness for one sampling batch.

    Fields, all on the sampling device:
        init_pos  (B, n_max, 3)      standard normal, initial coordinates
        init_unif (B, n_unif)        uniform, initial categorical state
        step_pos  (T, B, n_max, 3)   standard normal, per-step position noise
        step_unif (T, B, n_unif)     uniform, per-step categorical draws

    Index the per-step fields with ``t_int - 1``: the sampler visits
    ``t_int = T, T-1, ..., 1``, so slot 0 holds step 1 and slot T-1 holds step T.

    A molecule's slice depends only on (run_seed, mol_id) and on n_max. This is
    why n_max must be the GLOBAL padded size (29 for QM9 with explicit
    hydrogens) and never ``max(n_nodes)`` of the current batch: a batch-dependent
    n_max would make the draw sequence batch-dependent, which is the bug this
    module exists to remove.
    """

    __slots__ = ("init_pos", "init_unif", "step_pos", "step_unif", "n_max", "T",
                 "mol_ids", "run_seed")

    def __init__(self, init_pos, init_unif, step_pos, step_unif, n_max, T,
                 mol_ids, run_seed):
        self.init_pos = init_pos
        self.init_unif = init_unif
        self.step_pos = step_pos
        self.step_unif = step_unif
        self.n_max = n_max
        self.T = T
        self.mol_ids = list(mol_ids)
        self.run_seed = run_seed

    @classmethod
    def build(cls, run_seed: int, mol_ids, T: int, n_max: int, device="cpu",
              dtype=torch.float32):
        B = len(mol_ids)
        n_unif = n_unif_per_step(n_max)

        init_pos = torch.empty(B, n_max, 3, dtype=dtype)
        init_unif = torch.empty(B, n_unif, dtype=dtype)
        step_pos = torch.empty(T, B, n_max, 3, dtype=dtype)
        step_unif = torch.empty(T, B, n_unif, dtype=dtype)

        for b, mid in enumerate(mol_ids):
            init_pos[b] = torch.randn(n_max, 3, generator=_gen(run_seed, mid, P_INIT_POS), dtype=dtype)
            init_unif[b] = torch.rand(n_unif, generator=_gen(run_seed, mid, P_INIT_CAT), dtype=dtype)
            step_pos[:, b] = torch.randn(T, n_max, 3, generator=_gen(run_seed, mid, P_STEP_POS), dtype=dtype)
            step_unif[:, b] = torch.rand(T, n_unif, generator=_gen(run_seed, mid, P_STEP_CAT), dtype=dtype)

        return cls(init_pos.to(device), init_unif.to(device),
                   step_pos.to(device), step_unif.to(device),
                   n_max, T, mol_ids, run_seed)

    def step_slice(self, t_int: int):
        """(position noise, uniforms) for reverse step ``t_int``."""
        if not 1 <= t_int <= self.T:
            raise IndexError(f"t_int {t_int} outside [1, {self.T}]")
        k = t_int - 1
        return self.step_pos[k], self.step_unif[k]


# ----------------------------------------------------------------------------
# Tape-driven twin of diffusion_utils.sample_discrete_features
# ----------------------------------------------------------------------------

def sample_discrete_features_from_uniform(probX, probE, prob_charges, node_mask, unif):
    """Tape-driven twin of ``diffusion_utils.sample_discrete_features``.

    Masking, the upper-triangular restriction and the symmetrisation are
    identical to the original; only the three multinomial draws are replaced by
    inverse-CDF lookups on slices of ``unif``, of shape (B, 2*n + n*n).

    Returns the same ``(X_t, charges_t, E_t)`` triple as the original, as plain
    tensors, so the caller builds the PlaceHolder exactly as before.
    """
    bs, n = node_mask.shape
    expected = n_unif_per_step(n)
    if unif.shape != (bs, expected):
        raise ValueError(
            f"uniform slice has shape {tuple(unif.shape)}, expected {(bs, expected)}. "
            "The tape's n_max must equal the padded atom count of this batch; pad "
            "every batch to the global n_max."
        )

    # Masked rows must still define a distribution, as in the original.
    probX = probX.clone()
    prob_charges = prob_charges.clone()
    probE = probE.clone()
    probX[~node_mask] = 1 / probX.shape[-1]
    prob_charges[~node_mask] = 1 / prob_charges.shape[-1]

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask] = 1 / probE.shape[-1]

    uX = unif[:, :n]
    uC = unif[:, n:2 * n]
    uE = unif[:, 2 * n:].reshape(bs, n, n)

    X_t = categorical_from_uniform(probX, uX).squeeze(-1)                      # (bs, n)
    charges_t = categorical_from_uniform(prob_charges, uC).squeeze(-1)         # (bs, n)
    E_t = categorical_from_uniform(probE, uE).squeeze(-1)                      # (bs, n, n)

    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)
    return X_t, charges_t, E_t
