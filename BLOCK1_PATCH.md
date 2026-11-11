# Block 1 patch: per-molecule reproducible sampling

Three new files are added and two existing files are edited. Everything is
additive: with `mol_ids=None` the sampler behaves exactly as released, so the
diff is auditable against the Section 3.12 requirement that the patch "adds
capture and injection hooks and touches no parameter and no sampling step".

What the patch does NOT change: the distribution being sampled. Gaussian noise
is still standard normal; categorical draws move from `torch.multinomial(p, 1)`
to an inverse-CDF lookup on a uniform variate, which is distributionally
identical but a pure function of that variate. `tests/test_determinism.py`
checks the equivalence empirically.

## New files

| Path | Role |
|---|---|
| `midi/rng.py` | keyed counter-based RNG, the noise tape, the tape-driven categorical sampler |
| `midi/_compat.py` | the `torch.load` shim for torch >= 2.6 |
| `midi/test_pairing.py` | test T3, the end-to-end pairing gate |
| `tests/test_determinism.py` | tests T0-T2, no GPU or checkpoint needed |

## Edit 1 of 5 — `midi/main.py`, after line 2

```python
import torch
import midi._compat  # noqa: F401  -- torch.load default, must precede any load
import pytorch_lightning as pl
```

## Edit 2 of 5 — `midi/diffusion/noise_model.py`, imports (after line 7)

FIND
```python
import midi.utils as utils
from midi.diffusion import diffusion_utils
```
REPLACE
```python
import midi.utils as utils
from midi.diffusion import diffusion_utils
from midi import rng
```

## Edit 3 of 5 — `midi/diffusion/noise_model.py`, `sample_limit_dist` (line 244)

FIND
```python
    def sample_limit_dist(self, node_mask):
        """ Sample from the limit distribution of the diffusion process"""

        bs, n_max = node_mask.shape
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)

        U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_c = charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(node_mask.device)
        U_y = torch.zeros((bs, 0), device=node_mask.device)
```
REPLACE
```python
    def sample_limit_dist(self, node_mask, tape=None):
        """ Sample from the limit distribution of the diffusion process

        tape: optional midi.rng.NoiseTape. When supplied, every draw here is a
        pure function of (run_seed, molecule index) instead of the shared global
        stream, so molecule i is initialised identically in every arm.
        """
        bs, n_max = node_mask.shape
        dev = node_mask.device
        x_limit = self.X_marginals.expand(bs, n_max, -1)
        e_limit = self.E_marginals[None, None, None, :].expand(bs, n_max, n_max, -1)
        charges_limit = self.charges_marginals.expand(bs, n_max, -1)

        if tape is None:
            U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(dev)
            U_c = charges_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(dev)
            U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max).to(dev)
        else:
            assert tape.n_max == n_max, f"tape n_max {tape.n_max} != padded n {n_max}"
            u = tape.init_unif.to(dev)
            U_X = rng.categorical_from_uniform(x_limit.to(dev), u[:, :n_max]).squeeze(-1)
            U_c = rng.categorical_from_uniform(charges_limit.to(dev), u[:, n_max:2 * n_max]).squeeze(-1)
            U_E = rng.categorical_from_uniform(
                e_limit.to(dev), u[:, 2 * n_max:].reshape(bs, n_max, n_max)).squeeze(-1)
        U_y = torch.zeros((bs, 0), device=dev)
```

Then, further down in the same method, FIND
```python
        pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3, device=node_mask.device)
        pos = pos * node_mask.unsqueeze(-1)
```
REPLACE
```python
        if tape is None:
            pos = torch.randn(node_mask.shape[0], node_mask.shape[1], 3, device=node_mask.device)
        else:
            pos = tape.init_pos.to(node_mask.device)
        pos = pos * node_mask.unsqueeze(-1)
```

## Edit 4 of 5 — `midi/diffusion/noise_model.py`, `sample_zs_from_zt_and_pred` (line 279)

FIND
```python
    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling. """
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int
```
REPLACE
```python
    def sample_zs_from_zt_and_pred(self, z_t, pred, s_int, tape=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.

        tape: optional midi.rng.NoiseTape. Supplies this step's position noise
        and categorical uniforms from molecule i's own stream, so the whole
        trajectory is a pure function of (run_seed, molecule index).
        """
        bs, n, dxs = z_t.X.shape
        node_mask = z_t.node_mask
        t_int = z_t.t_int

        step_pos = step_unif = None
        if tape is not None:
            t_scalar = int(t_int.reshape(-1)[0].item())
            assert bool((t_int.reshape(-1) == t_scalar).all()), \
                "tape indexing assumes one t_int for the whole batch"
            step_pos, step_unif = tape.step_slice(t_scalar)
```

FIND
```python
        sampled_pos = torch.randn(z_t.pos.shape, device=z_t.pos.device) * node_mask.unsqueeze(-1)
```
REPLACE
```python
        if tape is None:
            sampled_pos = torch.randn(z_t.pos.shape, device=z_t.pos.device) * node_mask.unsqueeze(-1)
        else:
            sampled_pos = step_pos.to(z_t.pos.device) * node_mask.unsqueeze(-1)
```

FIND
```python
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c, node_mask=z_t.node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(sampled_s.charges, num_classes=self.charges_classes).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.E_classes).float()
```
REPLACE
```python
        if tape is None:
            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, prob_c,
                                                                 node_mask=z_t.node_mask)
            s_X, s_charges, s_E = sampled_s.X, sampled_s.charges, sampled_s.E
        else:
            s_X, s_charges, s_E = rng.sample_discrete_features_from_uniform(
                prob_X, prob_E, prob_c, node_mask=z_t.node_mask, unif=step_unif)

        X_s = F.one_hot(s_X, num_classes=self.X_classes).float()
        charges_s = F.one_hot(s_charges, num_classes=self.charges_classes).float()
        E_s = F.one_hot(s_E, num_classes=self.E_classes).float()
```

## Edit 5 of 5 — `midi/diffusion_model.py`

**5a, imports.** FIND
```python
from midi.diffusion.extra_features import ExtraFeatures
```
REPLACE
```python
from midi.diffusion.extra_features import ExtraFeatures
from midi import rng
```

**5b, `sample_batch` signature (line 367).** FIND
```python
    def sample_batch(self, n_nodes: list, number_chain_steps: int = 50, batch_id: int = 0, keep_chain: int = 0,
                     save_final: int = 0, test=True):
```
REPLACE
```python
    def sample_batch(self, n_nodes: list, number_chain_steps: int = 50, batch_id: int = 0, keep_chain: int = 0,
                     save_final: int = 0, test=True,
                     mol_ids=None, run_seed=None, n_max_fixed=None, visualize=True):
```

**5c, fixed padding and tape construction (lines 381-388).** FIND
```python
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask)
```
REPLACE
```python
        n_nodes = torch.Tensor(n_nodes).long().to(self.device)
        batch_size = len(n_nodes)
        # ---- Block 1: fixed padding and per-molecule noise tape ------------
        tape = None
        if mol_ids is not None:
            assert run_seed is not None, "mol_ids requires run_seed"
            assert len(mol_ids) == batch_size, \
                f"{len(mol_ids)} mol_ids for {batch_size} molecules"
            assert n_max_fixed is not None, (
                "pass n_max_fixed (29 for QM9 with explicit H). A batch-dependent "
                "n_max makes the tape batch-dependent, which is the bug this removes.")
            assert test or self.cfg.general.faster_sampling == 1, \
                "the tape is indexed by t_int, so faster_sampling must be 1"
            assert int(n_nodes.max()) <= n_max_fixed, \
                f"an atom count exceeds n_max_fixed={n_max_fixed}"
        n_max = n_max_fixed if n_max_fixed is not None else torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        if mol_ids is not None:
            tape = rng.NoiseTape.build(run_seed, mol_ids, self.T, n_max, device=self.device)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = self.noise_model.sample_limit_dist(node_mask=node_mask, tape=tape)
        # ---- end Block 1 ---------------------------------------------------
```

**5d, pass the tape down (line 404).** FIND
```python
            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array)
```
REPLACE
```python
            z_s = self.sample_zs_from_zt(z_t=z_t, s_int=s_array, tape=tape)
```

**5e, make visualisation optional (line 438).** FIND
```python
        if keep_chain > 0:
```
REPLACE
```python
        if visualize and keep_chain > 0:
```

**5f, same (lines 448-456).** FIND
```python
        if save_final > 0:
            self.print(f'Visualizing {save_final} individual molecules...')

        # Visualize the final molecules
        current_path = os.getcwd()
        result_path = os.path.join(current_path, f'graphs/epoch{self.current_epoch}_b{batch_id}/')
        _ = visualizer.visualize(result_path, molecule_list, num_molecules_to_visualize=save_final)
        self.print("Visualizing done.")
        return molecule_list
```
REPLACE
```python
        if visualize:
            if save_final > 0:
                self.print(f'Visualizing {save_final} individual molecules...')
            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path, f'graphs/epoch{self.current_epoch}_b{batch_id}/')
            _ = visualizer.visualize(result_path, molecule_list, num_molecules_to_visualize=save_final)
            self.print("Visualizing done.")
        return molecule_list
```

**5g, `sample_zs_from_zt` (line 458).** FIND
```python
    def sample_zs_from_zt(self, z_t, s_int):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int)
        return z_s
```
REPLACE
```python
    def sample_zs_from_zt(self, z_t, s_int, tape=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        extra_data = self.extra_features(z_t)
        pred = self.forward(z_t, extra_data)
        z_s = self.noise_model.sample_zs_from_zt_and_pred(z_t=z_t, pred=pred, s_int=s_int, tape=tape)
        return z_s
```

## Run order

```powershell
cd C:\Users\tsuma.thomas\Documents\MiDi-ActivationSteering

# T0-T2: seconds, no GPU, no checkpoint
python tests\test_determinism.py

# T3: the gate. CPU first, because it is the bit-exact one.
cd midi
python test_pairing.py dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive general.wandb=disabled general.gpus=0  "general.test_only=C:/Users/tsuma.thomas/Documents/MiDi-ActivationSteering/midi/checkpoints/checkpoint_qm9_h_adaptive.ckpt"

# then the same with general.gpus=1
```

T3 on CPU runs 8 molecules plus 2 through 500 denoising steps each, so expect
several minutes. It is the gate: do not start Block 3 until it passes.

## Record for the manifest

`run_seed`, `n_max` (29), `T` (500), the git commit of this patch, and the T3
verdict including the CUDA max-deviation figure if the GPU run is not bit-exact.
