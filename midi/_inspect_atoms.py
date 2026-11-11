import hydra, omegaconf, numpy as np, torch
import midi._compat  # noqa
from midi.datasets import qm9_dataset

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg):
    dm = qm9_dataset.QM9DataModule(cfg)
    di = qm9_dataset.QM9infos(datamodule=dm, cfg=cfg)
    ds = dm.test_dataloader().dataset
    print("len(test dataset) =", len(ds))
    print("atom_decoder      =", di.atom_decoder)
    print("atom_encoder      =", getattr(di, "atom_encoder", None))
    d0 = ds[0]
    print("keys              =", sorted(k for k in d0.keys))
    for name in ("x", "pos", "charges", "edge_attr", "y"):
        v = getattr(d0, name, None)
        if v is None:
            print(f"  {name:10s} absent")
        else:
            print(f"  {name:10s} shape={tuple(v.shape)} dtype={v.dtype} numel={v.numel()}")
    print("x[:12] =", d0.x[:12].tolist())
    print("smiles =", getattr(d0, "smiles", None))
    print()
    for j in (0, 1, 2):
        x = ds[j].x
        if x.dim() == 2 and x.shape[1] == len(di.atom_decoder):
            idx = x.argmax(dim=1)
            kind = "one-hot"
        else:
            idx = x.reshape(-1).long()
            kind = "index"
        cnt = np.bincount(idx.numpy(), minlength=len(di.atom_decoder))
        pretty = {di.atom_decoder[k]: int(cnt[k]) for k in range(len(cnt))}
        print(f"mol {j}: encoding={kind} n_atoms(x.numel)={x.numel()} len(idx)={len(idx)} counts={pretty} sum={int(cnt.sum())} smiles={getattr(ds[j], chr(115)+chr(109)+chr(105)+chr(108)+chr(101)+chr(115), None)}")

main()
