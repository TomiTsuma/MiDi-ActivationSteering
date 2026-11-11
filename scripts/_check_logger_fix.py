"""Verify the stage-10 fix at the exact code path that failed.

The traceback went:
    trainer.test -> _run -> _log_hyperparams(self)
      -> logger.log_hyperparams -> TensorBoardLogger.log_metrics
      -> self.experiment -> import torch.utils.tensorboard
      -> tensorboard -> tensorflow -> googleapiclient -> oauth2client
      -> pyOpenSSL -> AttributeError: module 'lib' has no attribute 'GEN_EMAIL'
    re-raised by Lightning as: ValueError: you tried to log -1

_log_hyperparams iterates trainer.loggers. With logger=False that list is empty,
so the whole chain is never entered. This checks exactly that, without loading
the dataset, the checkpoint or the GPU.
"""
import sys

import torch  # noqa: F401
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.utilities import _log_hyperparams

print("torch           ", torch.__version__)

print("\n--- confirm the underlying breakage is real ---")
try:
    import OpenSSL.crypto  # noqa: F401
    print("  pyOpenSSL imports cleanly (so the root cause may be gone)")
except Exception as e:
    print(f"  pyOpenSSL BROKEN: {type(e).__name__}: {e}")
    print("  -> this is why the TensorBoard logger could not be created")

print("\n--- default Trainer (what main.py used to build) ---")
t_default = Trainer(accelerator="cpu", devices=1, logger=True,
                    enable_progress_bar=False, max_epochs=1)
print(f"  trainer.loggers = {[type(l).__name__ for l in t_default.loggers]}")

print("\n--- patched Trainer (logger=False) ---")
t_fixed = Trainer(accelerator="cpu", devices=1, logger=False,
                  enable_progress_bar=False, max_epochs=1)
print(f"  trainer.loggers = {[type(l).__name__ for l in t_fixed.loggers]}")

if len(t_fixed.loggers) != 0:
    print("\nFAIL: logger=False did not clear trainer.loggers")
    sys.exit(1)

print("\n--- call the function that raised, on the patched trainer ---")
try:
    _log_hyperparams(t_fixed)
    print("  _log_hyperparams(trainer) returned cleanly")
except Exception as e:
    print(f"  FAIL: still raises {type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- and on the default trainer, to show it is the logger at fault ---")
try:
    _log_hyperparams(t_default)
    print("  _log_hyperparams also fine with the default logger")
    print("  (so this environment no longer reproduces the original failure)")
except Exception as e:
    print(f"  raises {type(e).__name__}: {str(e).strip()[:90]}")
    print("  <- exactly the stage-10 failure, confirmed reproducible")

print("\nPASS: logger=False removes the failing path.")
