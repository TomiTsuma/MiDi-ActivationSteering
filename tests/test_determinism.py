"""Block 1 unit tests: per-molecule randomness is batch-independent.

Run from the repository root, no GPU and no checkpoint needed:

    python -m pytest tests/test_determinism.py -v
    # or, without pytest:
    python tests/test_determinism.py

These cover the tape and the categorical substitution. The end-to-end pairing
test, which needs the checkpoint, is midi/test_pairing.py (test T3).
"""

import torch

from midi.rng import (
    NoiseTape,
    categorical_from_uniform,
    derive_seed,
    n_unif_per_step,
    sample_n_nodes,
    tape_bytes,
)

T_SMALL = 8          # stand-in for the 500 real steps
N_MAX = 29           # QM9 with explicit hydrogens; must be the GLOBAL value
SEED = 1234


# --------------------------------------------------------------------------
# T0: key derivation
# --------------------------------------------------------------------------

def test_derive_seed_is_deterministic_and_distinct():
    a = derive_seed(SEED, 7, "init_pos")
    b = derive_seed(SEED, 7, "init_pos")
    assert a == b, "same key must give the same seed within a process"
    assert a != derive_seed(SEED, 8, "init_pos"), "molecule index must matter"
    assert a != derive_seed(SEED + 1, 7, "init_pos"), "run seed must matter"
    assert a != derive_seed(SEED, 7, "step_pos"), "purpose must matter"
    assert 0 <= a < 2 ** 63


def test_derive_seed_is_stable_across_processes():
    """blake2b, not Python's randomised hash(). These literals were produced by
    this same function; if they ever change, the tape is no longer portable and
    every stored run becomes unreproducible."""
    assert derive_seed(0, 0, "init_pos") == derive_seed(0, 0, "init_pos")
    import subprocess, sys, json
    out = subprocess.run(
        [sys.executable, "-c",
         "import hashlib,json;"
         "k=b'0|0|init_pos';"
         "print(json.dumps(int.from_bytes(hashlib.blake2b(k,digest_size=8).digest(),'big')&((1<<63)-1)))"],
        capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == derive_seed(0, 0, "init_pos")


# --------------------------------------------------------------------------
# T_dist: the categorical substitution is distributionally correct
# --------------------------------------------------------------------------

def test_categorical_from_uniform_matches_multinomial():
    torch.manual_seed(0)
    probs = torch.tensor([0.1, 0.5, 0.25, 0.15])
    n = 400_000

    u = torch.rand(n)
    got = categorical_from_uniform(probs.expand(n, -1), u).squeeze(-1)
    emp = torch.bincount(got, minlength=4).float() / n

    ref = torch.multinomial(probs.expand(n, -1), 1).squeeze(-1)
    emp_ref = torch.bincount(ref, minlength=4).float() / n

    assert torch.allclose(emp, probs, atol=3e-3), f"inverse CDF empirical {emp} vs {probs}"
    assert torch.allclose(emp, emp_ref, atol=5e-3), f"{emp} vs multinomial {emp_ref}"


def test_categorical_never_selects_zero_probability():
    probs = torch.tensor([0.4, 0.0, 0.6]).expand(10_000, -1)
    u = torch.rand(10_000)
    got = categorical_from_uniform(probs, u).squeeze(-1)
    assert (got != 1).all(), "a zero-probability category was selected"


def test_categorical_endpoints():
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    assert categorical_from_uniform(probs, torch.tensor([0.0])).item() == 0
    assert categorical_from_uniform(probs, torch.tensor([0.999999])).item() == 3


# --------------------------------------------------------------------------
# T1: same key, same randomness
# --------------------------------------------------------------------------

def test_tape_is_reproducible_across_builds():
    ids = [0, 1, 2, 3]
    a = NoiseTape.build(SEED, ids, T_SMALL, N_MAX)
    b = NoiseTape.build(SEED, ids, T_SMALL, N_MAX)
    for f in ("init_pos", "init_unif", "step_pos", "step_unif"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f"{f} not reproducible"


def test_tape_differs_across_run_seeds():
    a = NoiseTape.build(SEED, [0], T_SMALL, N_MAX)
    b = NoiseTape.build(SEED + 1, [0], T_SMALL, N_MAX)
    assert not torch.equal(a.init_pos, b.init_pos)


# --------------------------------------------------------------------------
# T2: the real test. Batch composition must not matter.
# --------------------------------------------------------------------------

def test_tape_slice_is_batch_independent():
    """Molecule 4 must get identical randomness in a batch of 8, in a batch of
    2, and alone, at whatever position it occupies. This is the test a shared
    RNG stream fails."""
    big = NoiseTape.build(SEED, [0, 1, 2, 3, 4, 5, 6, 7], T_SMALL, N_MAX)
    small = NoiseTape.build(SEED, [4, 5], T_SMALL, N_MAX)
    alone = NoiseTape.build(SEED, [4], T_SMALL, N_MAX)
    reordered = NoiseTape.build(SEED, [5, 4], T_SMALL, N_MAX)

    for name, tape, pos in [("batch of 2", small, 0),
                            ("alone", alone, 0),
                            ("reordered", reordered, 1)]:
        assert torch.equal(big.init_pos[4], tape.init_pos[pos]), f"init_pos, {name}"
        assert torch.equal(big.init_unif[4], tape.init_unif[pos]), f"init_unif, {name}"
        assert torch.equal(big.step_pos[:, 4], tape.step_pos[:, pos]), f"step_pos, {name}"
        assert torch.equal(big.step_unif[:, 4], tape.step_unif[:, pos]), f"step_unif, {name}"


def test_n_nodes_is_batch_independent():
    prob = torch.zeros(30)
    for k, c in {19: 13531, 18: 12970, 17: 12691, 16: 10332, 21: 9889,
                 20: 9194, 15: 7700, 14: 5102, 23: 4765, 22: 3324}.items():
        prob[k] = c
    prob = prob / prob.sum()

    big = sample_n_nodes(SEED, list(range(8)), prob)
    small = sample_n_nodes(SEED, [4, 5], prob)
    assert big[4].item() == small[0].item()
    assert big[5].item() == small[1].item()
    assert (big >= 3).all() and (big <= 29).all()


def test_step_slice_indexing():
    tape = NoiseTape.build(SEED, [0, 1], T_SMALL, N_MAX)
    p1, u1 = tape.step_slice(1)
    assert torch.equal(p1, tape.step_pos[0]) and torch.equal(u1, tape.step_unif[0])
    pT, uT = tape.step_slice(T_SMALL)
    assert torch.equal(pT, tape.step_pos[T_SMALL - 1])
    for bad in (0, T_SMALL + 1):
        try:
            tape.step_slice(bad)
        except IndexError:
            pass
        else:
            raise AssertionError(f"t_int={bad} should raise")


def test_uniform_budget_matches_sampler():
    assert n_unif_per_step(29) == 2 * 29 + 29 * 29 == 899


def test_tape_sizing_formula():
    """Printed for Block 5: the tape trades against GPU memory."""
    for B in (16, 32, 64, 128, 256):
        mb = tape_bytes(B, 500, 29) / 1024 ** 2
        print(f"  batch {B:4d} -> tape {mb:8.1f} MiB")
    assert tape_bytes(64, 500, 29) / 1024 ** 2 < 200


if __name__ == "__main__":
    import sys, traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
