"""Runs INSIDE the MOKIT conda env (python 3.12), not the main workflow env.

MOKIT's conda builds stop at python 3.12 while the workflow runs on 3.13, so
the orbital conversion is isolated behind a subprocess boundary. This script
is the whole of that boundary: it reads a Gaussian-format checkpoint with
MOKIT's `fch2py` and writes the MO coefficients as a plain .npy, which the
main env loads with no MOKIT dependency at all.

Invoked by mcxray/orbital_io.py; not meant to be run by hand.

    conda run -n <env> python _mokit_bridge.py <file.fch> <out.npy> [a|b]
"""
import sys

import numpy as np
from mokit.lib.fch2py import fch2py


def fch_dims(fch):
    """Read (nbf, nif) out of the fch header."""
    nbf = nif = None
    with open(fch) as f:
        for line in f:
            if line.startswith('Number of basis functions'):
                nbf = int(line.split()[-1])
            elif line.startswith('Number of independent functions'):
                nif = int(line.split()[-1])
            if nbf is not None and nif is not None:
                break
    if nbf is None or nif is None:
        raise RuntimeError(f'could not read nbf/nif from {fch}')
    return nbf, nif


if __name__ == '__main__':
    fch, out = sys.argv[1], sys.argv[2]
    ab = sys.argv[3] if len(sys.argv) > 3 else 'a'
    nbf, nif = fch_dims(fch)
    mo = fch2py(fch, nbf, nif, ab)
    np.save(out, np.asarray(mo))
    print(f'fch2py: {fch} -> {out}  shape={np.shape(mo)}  (nbf={nbf}, nif={nif})')
