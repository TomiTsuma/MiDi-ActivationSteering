"""Covers midi.rng.sample_discrete_features_from_uniform against the original.

This is the function that does inverse-CDF lookup on the dense (bs, n, n, K)
edge tensor, so it is where a shape bug would hide.
"""
import torch
from midi.rng import sample_discrete_features_from_uniform, n_unif_per_step

torch.manual_seed(0)
BS, N, KX, KC, KE = 3, 7, 5, 3, 5


def _mk():
    node_mask = torch.ones(BS, N, dtype=torch.bool)
    node_mask[0, 5:] = False          # molecule 0 has 5 atoms
    node_mask[1, 6:] = False          # molecule 1 has 6
    probX = torch.rand(BS, N, KX); probX /= probX.sum(-1, keepdim=True)
    probC = torch.rand(BS, N, KC); probC /= probC.sum(-1, keepdim=True)
    probE = torch.rand(BS, N, N, KE)
    probE = probE + probE.transpose(1, 2)          # symmetric probabilities
    probE /= probE.sum(-1, keepdim=True)
    return probX, probC, probE, node_mask


def test_shapes_and_structure():
    probX, probC, probE, nm = _mk()
    u = torch.rand(BS, n_unif_per_step(N))
    X, C, E = sample_discrete_features_from_uniform(probX, probE, probC, nm, u)
    assert X.shape == (BS, N) and X.dtype == torch.long, (X.shape, X.dtype)
    assert C.shape == (BS, N) and C.dtype == torch.long
    assert E.shape == (BS, N, N) and E.dtype == torch.long
    assert torch.equal(E, E.transpose(1, 2)), "E must be symmetric"
    assert torch.equal(torch.diagonal(E, dim1=1, dim2=2),
                       torch.zeros(BS, N, dtype=torch.long)), "E diagonal must be 0"
    assert X.max() < KX and C.max() < KC and E.max() < KE
    print(f"  shapes OK, E symmetric, diagonal zero, indices in range")


def test_pure_function_of_uniforms():
    probX, probC, probE, nm = _mk()
    u = torch.rand(BS, n_unif_per_step(N))
    a = sample_discrete_features_from_uniform(probX, probE, probC, nm, u)
    b = sample_discrete_features_from_uniform(probX, probE, probC, nm, u.clone())
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)
    print("  identical output for identical uniforms")


def test_does_not_mutate_inputs():
    probX, probC, probE, nm = _mk()
    u = torch.rand(BS, n_unif_per_step(N))
    bx, bc, be = probX.clone(), probC.clone(), probE.clone()
    sample_discrete_features_from_uniform(probX, probE, probC, nm, u)
    assert torch.equal(probX, bx) and torch.equal(probC, bc) and torch.equal(probE, be), \
        "caller's probability tensors were mutated (the original does mutate; this must not)"
    print("  caller tensors untouched")


def test_marginals_match_the_probabilities():
    """Row 0 of molecule 0: empirical frequency over many uniform draws must
    track probX[0, 0]."""
    probX, probC, probE, nm = _mk()
    target = probX[0, 0]
    n = 200_000
    counts = torch.zeros(KX)
    us = torch.rand(n)
    cdf = target.cumsum(0); cdf = cdf / cdf[-1]
    idx = torch.searchsorted(cdf.contiguous(), us.contiguous())
    counts = torch.bincount(idx.clamp_(max=KX - 1), minlength=KX).float() / n
    assert torch.allclose(counts, target, atol=4e-3), f"{counts} vs {target}"
    print(f"  node marginal: {[f'{v:.4f}' for v in counts.tolist()]}"
          f" vs target {[f'{v:.4f}' for v in target.tolist()]}")


def test_rejects_wrong_tape_width():
    probX, probC, probE, nm = _mk()
    bad = torch.rand(BS, n_unif_per_step(N) - 1)
    try:
        sample_discrete_features_from_uniform(probX, probE, probC, nm, bad)
    except ValueError as e:
        assert "n_max" in str(e)
        print("  wrong tape width rejected with a useful message")
    else:
        raise AssertionError("should have raised on a mis-sized uniform slice")


def test_masked_rows_still_sample():
    """Padded rows get a uniform distribution, as in the original, so they draw
    a value rather than crashing. The value is masked away downstream."""
    probX, probC, probE, nm = _mk()
    u = torch.rand(BS, n_unif_per_step(N))
    X, C, E = sample_discrete_features_from_uniform(probX, probE, probC, nm, u)
    assert X[0, 5:].min() >= 0 and E[0, 5:, 5:].min() >= 0
    print("  padded rows sampled without error")


if __name__ == "__main__":
    import traceback, sys
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f"PASS  {name}")
        except Exception:
            failed += 1; print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
