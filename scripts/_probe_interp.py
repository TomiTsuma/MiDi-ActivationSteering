"""Report this interpreter's identity, deps and midi importability. One line each."""
import sys

print("VER|%s|%s" % (sys.version.split()[0], sys.executable))

try:
    import torch
    import torch_geometric  # noqa: F401
    import rdkit            # noqa: F401
    import hydra            # noqa: F401
    import numpy            # noqa: F401
    print("DEPS|ok|torch %s|cuda %s" % (torch.__version__, torch.cuda.is_available()))
except Exception as e:
    print("DEPS|missing|%s: %s" % (type(e).__name__, e))

try:
    import midi             # noqa: F401
    import midi.rng         # noqa: F401
    import midi.capture     # noqa: F401
    import midi.verify_store  # noqa: F401
    import midi.qm9_labels  # noqa: F401
    print("MIDI|ok")
except Exception as e:
    print("MIDI|missing|%s: %s" % (type(e).__name__, e))
