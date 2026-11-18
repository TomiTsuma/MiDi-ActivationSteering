"""Run GFN2/PTB xtb on one molecule and parse the polarisability.

Which number, and why it matters
--------------------------------
xtb can report TWO things called polarisability and they are not
interchangeable:

  Mol. alpha(0) /au    printed by `--gfn 2 --verbose` in the dispersion
                       section. It is the D4 model's sum of CHARGE-SCALED
                       ATOMIC polarisabilities. Verified on water:
                       O 6.657 + H 1.386 + H 1.386 = 9.429 = the printed
                       total, exactly. It is ADDITIVE BY CONSTRUCTION, so it
                       is a composition-and-charge sum, not a response
                       property. Using it as the "independent physical check"
                       would rebuild the size confound the check exists to
                       test. Do not use it.

  --ptb --alpha        a NUMERICAL polarisability tensor from the PTB
                       response, with a total isotropic dipole
                       polarisability. This is a genuine field-response
                       property and the right comparator for QM9's DFT alpha.
                       Water: 7.2126 a.u.

Units: a.u. of polarisability is Bohr^3, the same unit QM9 reports alpha in,
so the two are directly comparable with no conversion.

Isolation: xtb writes xtbrestart, charges, wbo and xtbtopo.mol into its working
directory, and will READ a stale xtbrestart as an SCF guess. That would make
one molecule's result depend on whichever molecule ran before it in the same
directory. Every call therefore runs in its own directory and passes
--norestart.
"""

import os
import re
import shutil
import subprocess
import tempfile

ATOM_DECODER = ("H", "C", "N", "O", "F")
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


def uhf_for(symbols, charge):
    """Unpaired-electron count to pass as --uhf: 1 for an odd electron count.

    PTB assumes a closed-shell singlet and aborts with "Number of electrons and
    spin multiplicity do not match" when the electron count is odd. A nonzero
    net formal charge makes that happen often, and MiDi does generate charged
    graphs, so without this every ion fails. That would be a CORRELATED
    dropout: if steered arms produce more ions than the control, dropping them
    silently biases the arm comparison in whichever direction ions differ.
    Setting --uhf from the parity keeps them in.
    """
    n_elec = sum(Z[s] for s in symbols) - int(charge)
    return n_elec % 2
_XTB_HOME = os.environ.get("XTB_DIST", "/home/claude/xtbdl/xtb-dist")
_HDR = re.compile(r"Total isotropic dipole polarizability")


def xtb_env(dist=None):
    dist = dist or _XTB_HOME
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(dist, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["XTBPATH"] = os.path.join(dist, "share", "xtb")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env, os.path.join(dist, "bin", "xtb")


def write_xyz(path, symbols, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            f.write(f"{s:<3s} {x: .8f} {y: .8f} {z: .8f}\n")


def ptb_alpha(symbols, coords, dist=None, timeout=120, opt=False, charge=0,
              uhf=None):
    """(alpha_au, alpha_ang3, status). alpha is None unless status == 'ok'.

    opt=True relaxes the geometry with GFN2 first (`--opt`), then scores the
    relaxed structure. opt=False scores the coordinates exactly as given.
    """
    env, exe = xtb_env(dist)
    if uhf is None:
        uhf = uhf_for(symbols, charge)
    d = tempfile.mkdtemp(prefix="xtb_")
    try:
        xyz = os.path.join(d, "m.xyz")
        write_xyz(xyz, symbols, coords)
        if opt:
            r = subprocess.run([exe, "m.xyz", "--gfn", "2", "--opt", "tight",
                                "--norestart", "--chrg", str(charge),
                                "--uhf", str(uhf)],
                               cwd=d, env=env, capture_output=True, text=True,
                               timeout=timeout)
            relaxed = os.path.join(d, "xtbopt.xyz")
            if r.returncode != 0 or not os.path.exists(relaxed):
                return None, None, "opt_failed"
            xyz = relaxed
        r = subprocess.run([exe, os.path.basename(xyz), "--ptb", "--alpha",
                            "--norestart", "--chrg", str(charge),
                            "--uhf", str(uhf)],
                           cwd=d, env=env, capture_output=True, text=True,
                           timeout=timeout)
        lines = r.stdout.splitlines()
        for i, ln in enumerate(lines):
            if _HDR.search(ln):
                for nxt in lines[i + 1:i + 3]:
                    nums = re.findall(r"[-+]?\d+\.\d+", nxt)
                    if len(nums) >= 2:
                        return float(nums[0]), float(nums[1]), "ok"
        if r.returncode != 0:
            return None, None, "xtb_nonzero"
        return None, None, "no_alpha_in_output"
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    except Exception as e:
        return None, None, f"error:{type(e).__name__}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
