"""Compatibility shim for torch >= 2.6.

torch 2.6 changed the default of ``torch.load``'s ``weights_only`` argument
from False to True. That default rejects two things this repository must load:

  * the processed PyG dataset, which contains
    ``torch_geometric.data.data.DataEdgeAttr`` (qm9_dataset.py:59,
    geom_dataset.py:38);
  * the Lightning checkpoint, whose ``hyper_parameters`` hold an OmegaConf
    DictConfig. Lightning 2.0.4 calls ``torch.load`` without passing
    ``weights_only``, so it inherits the new default.

Every file loaded here is either produced locally by this repository or is the
released MiDi checkpoint, obtained deliberately, so full unpickling is intended.
This does disable the ``weights_only`` protection process-wide; that protection
guards against arbitrary code execution from untrusted checkpoints, which is a
risk already accepted for these specific files.

Import this before any torch.load happens.
"""

import torch

_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


if getattr(torch.load, "__module__", None) != __name__:
    _load.__module__ = __name__
    torch.load = _load
