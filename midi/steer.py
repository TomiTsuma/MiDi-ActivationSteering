"""midi/steer.py -- training-free activation steering during reverse diffusion.

Hooks only. No parameter is touched, no sampling step is edited, so the patch
diff is the whole record of the intervention, which is what Section 3.12 asks
for.

How the hook learns the timestep, which was the one real unknown:

  sample_zs_from_zt calls `self.forward(z_t, extra_data)` DIRECTLY
  (diffusion_model.py:479). A direct method call bypasses nn.Module.__call__,
  so any hook registered on the LightningModule never fires. But forward()
  ends with `self.model(model_input)` (line 552), a proper call, and it builds
  `model_input.y = hstack((z_t.y, extra_data.y, z_t.t))`.

  Measured on this checkpoint: a forward pre-hook on model.model fires exactly
  500 times per sample_batch, model_input.y has shape (bs, 1), and its single
  column runs 1.0 down to 0.002. So t_int = round(y * T) recovers 500, 499,
  ..., 1 exactly.

  A plain call counter gives the same answer, and the pre-hook asserts the two
  agree. If the sampling loop is ever changed so that they diverge, this fails
  loudly instead of silently steering the wrong timesteps.
"""
import numpy as np
import torch

from midi.capture import pool_nodes, pool_edges


class SteerState:
    """Everything the hooks read, and where they write their diagnostics.

    coeff is in units of ||d||, so the injected vector is coeff * d_norm * v
    and coeff = 1 moves the molecule by the full low-to-high gap.

    window is an inclusive timestep range (t_lo, t_hi) in the model's own
    numbering, where t = T is the noisiest step and t = 1 the last. So
    (50, 100) injects over the final stretch and (200, 300) earlier.
    """

    def __init__(self, layer, channel, v, d_norm, coeff, window, T,
                 device, dtype=torch.float32, record=True, force_apply=False,
                 v_timesteps=None):
        # force_apply exists for ONE test: run the add path with scale 0 and
        # require the output to stay bit-identical. Without it, coeff=0 is
        # short-circuited and proves only that attaching hooks is harmless,
        # not that the arithmetic is a true no-op. A dtype promotion, a device
        # copy or a mask that touches padding would all survive the weaker
        # test and corrupt every steered run.
        self.force_apply = bool(force_apply)
        assert channel in ("X", "E", "y"), channel
        self.layer = int(layer)
        self.channel = channel

        # v may be a single direction (d,) or one per captured timestep
        # (n_t, d) with v_timesteps. The per-timestep form matters: measured
        # on this store, directions at ADJACENT captured timesteps have cosine
        # about 0.99, but across the full range from t=500 to t=50 it falls to
        # -0.07. Holding one direction over a wide window would be steering
        # along something the later steps do not share.
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            self.V = torch.as_tensor(v[None, :], dtype=dtype, device=device)
            self.v_ts = np.array([-1])
            self.D = np.array([float(d_norm)])
        else:
            if v_timesteps is None or len(v_timesteps) != v.shape[0]:
                raise ValueError("per-timestep directions need v_timesteps")
            self.V = torch.as_tensor(v, dtype=dtype, device=device)
            self.v_ts = np.asarray(v_timesteps, dtype=int)
            dn = np.asarray(d_norm, dtype=np.float64)
            self.D = np.full(v.shape[0], float(dn)) if dn.ndim == 0 else dn
        self.coeff = float(coeff)
        self.t_lo, self.t_hi = int(min(window)), int(max(window))
        self.T = int(T)
        self.record = record
        self.reset()

    def reset(self):
        self.t_now = None
        self.n_calls = 0
        self.n_injected = 0
        self.steps = []          # one row per injected step (batch means)
        self._per_mol = None     # running sums, per molecule

    def current(self):
        """(v, scale) for the step in flight, picking the direction captured
        nearest the current timestep."""
        if len(self.v_ts) == 1:
            i = 0
        else:
            i = int(np.argmin(np.abs(self.v_ts - int(self.t_now))))
        return self.V[i], self.coeff * float(self.D[i])

    @property
    def scale(self):
        """Scale at the reference direction; for tests and reporting."""
        return self.coeff * float(self.D[0])

    def active(self):
        in_window = (self.t_now is not None
                     and self.t_lo <= self.t_now <= self.t_hi)
        return in_window and (self.force_apply or self.coeff != 0.0)

    def window_size(self):
        return self.t_hi - self.t_lo + 1

    def per_molecule(self):
        """Mean rho and cos over the injected steps, per molecule."""
        if self._per_mol is None:
            return None
        s_rho, s_cos, k = self._per_mol
        return (s_rho / max(k, 1)).cpu().numpy(), (s_cos / max(k, 1)).cpu().numpy()


def make_t_prehook(state):
    """Reads the timestep off model_input.y and parks it on the state."""
    def pre_hook(module, args):
        mi = args[0]
        # y is (bs, k) with the normalised time as its LAST column.
        t_norm = mi.y[0, -1].detach().float().item()
        t_from_y = int(round(t_norm * state.T))
        t_from_count = state.T - state.n_calls
        if t_from_y != t_from_count:
            raise RuntimeError(
                f"timestep disagreement at call {state.n_calls}: y says "
                f"{t_from_y}, the call counter says {t_from_count}. The "
                f"sampling loop is not what steer.py assumes; fix this before "
                f"trusting any steered run.")
        state.t_now = t_from_y
        state.n_calls += 1
        return None
    return pre_hook


def _masked_mean(X, node_mask):
    """The X_mean block exactly as pool_nodes computes it."""
    m = node_mask.unsqueeze(-1).to(X.dtype)
    return (X * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def make_inject_hook(state):
    """Adds coeff * d_norm * v to the layer output, inside the window only.

    Diagnostics, per Section 3.9: the coefficient moves the ANGLE and the NORM
    at once, and a plateau in the outcome could be either a saturating angle
    or norm-driven damage. Both are recorded so they can be told apart.

        rho   = ||a'|| / ||a||          how much bigger the activation got
        cos   = (a' . v) / ||a'||       how aligned it is with the direction

    measured on the pooled X_mean, the same space the direction lives in.
    """
    def hook(module, inputs, output):
        if not state.active():
            return None
        v, s = state.current()
        nm = output.node_mask

        if state.channel == "X":
            a = _masked_mean(output.X, nm)
            output.X = output.X + s * v.view(1, 1, -1) * nm.unsqueeze(-1)
            a2 = _masked_mean(output.X, nm)
        elif state.channel == "y":
            a = output.y
            output.y = output.y + s * v
            a2 = output.y
        else:  # E
            em = (nm.unsqueeze(-1).unsqueeze(2) * nm.unsqueeze(-1).unsqueeze(1))
            a = (output.E * em.to(output.E.dtype)).sum(dim=(1, 2)) \
                / em.to(output.E.dtype).sum(dim=(1, 2)).clamp_min(1.0)
            output.E = output.E + s * v.view(1, 1, 1, -1) * em.to(output.E.dtype)
            output.E = 0.5 * (output.E + output.E.transpose(1, 2))
            a2 = (output.E * em.to(output.E.dtype)).sum(dim=(1, 2)) \
                / em.to(output.E.dtype).sum(dim=(1, 2)).clamp_min(1.0)

        if state.record:
            with torch.no_grad():
                na = a.norm(dim=-1).clamp_min(1e-12)
                na2 = a2.norm(dim=-1).clamp_min(1e-12)
                rho = na2 / na
                cos = (a2 @ v) / na2
                if state._per_mol is None:
                    state._per_mol = [torch.zeros_like(rho),
                                      torch.zeros_like(cos), 0]
                state._per_mol[0] += rho
                state._per_mol[1] += cos
                state._per_mol[2] += 1
                state.steps.append({
                    "t": state.t_now,
                    "rho_mean": float(rho.mean()),
                    "cos_mean": float(cos.mean()),
                    "norm_unsteered_mean": float(na.mean()),
                })
        state.n_injected += 1
        return output
    return hook


def attach(model, state):
    """Register both hooks. Returns handles; pass them to detach()."""
    inner = model.model
    if not hasattr(inner, "tf_layers"):
        raise RuntimeError("model.model has no tf_layers; the architecture "
                           "is not what steer.py assumes")
    return [
        inner.register_forward_pre_hook(make_t_prehook(state)),
        inner.tf_layers[state.layer].register_forward_hook(make_inject_hook(state)),
    ]


def detach(handles):
    for h in handles:
        h.remove()
