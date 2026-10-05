"""Calculated L-edge XAS (step 6) against a measured spectrum.

  python scripts/plot_xas_vs_experiment.py --system=crddpd \
      --exp=PATH [--tags=,tPBE0] [--gamma=0.2]

--tags lists the spectra{_TAG}.npz files to plot ('' is the untagged tPBE
run). The experiment (two columns: eV, intensity) has its pre-edge level
subtracted; every trace is normalized to its L3 maximum and each
calculated trace is shifted rigidly so its L3 maximum sits on the
measured one. The shifts are printed and written in the legend.

--gamma=EV rebroadens the step-6 sticks with that Lorentzian half width
instead of step 6's (common.GAMMA_CORE_EV); the plot is then saved as
xas_vs_experiment_gEV.png.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import systems

PRE_EDGE_EV = 3.0        # width of the pre-edge window used as baseline
OFFSET = 0.6             # vertical offset between traces
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR, METAL = cfg['run_dir'], cfg['metal']
    exp_path, tags, gamma = None, ['', 'tPBE0'], None
    for a in sys.argv[1:]:
        if a.startswith('--exp='):
            exp_path = a.split('=', 1)[1]
        elif a.startswith('--tags='):
            tags = a.split('=', 1)[1].split(',')
        elif a.startswith('--gamma='):
            gamma = float(a.split('=', 1)[1])
    if exp_path is None:
        raise SystemExit('--exp=PATH is required')

    x, y = np.loadtxt(exp_path, unpack=True)
    y = y - y[x < x.min() + PRE_EDGE_EV].mean()
    y /= y.max()
    x_l3 = x[y.argmax()]
    print(f'experiment: {os.path.basename(exp_path)}, {x.min():.1f}-'
          f'{x.max():.1f} eV, L3 maximum at {x_l3:.2f} eV')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    top = OFFSET * len(tags)
    ax.plot(x, y + top, color=INK, lw=2, label='experiment')
    ax.text(x.max(), top + y[-1] + 0.06, 'experiment', color=INK,
            ha='right', va='bottom', fontsize=9)
    for n, (tag, col) in enumerate(zip(tags, COLORS)):
        s = np.load(os.path.join(RUN_DIR,
                                 f'spectra{"_" + tag if tag else ""}.npz'))
        g, sp = s['grid'], s['spectrum']
        if gamma is not None:
            sp = common.broaden_sticks(s['de_ev'], s['osc'], g, gamma)
        # L3 maximum: the highest point below the sticks' intensity
        # centroid, which lies between the L3 and L2 branches. (With
        # narrow lines the global maximum can be the L2 peak.)
        l3 = g < np.average(s['de_ev'], weights=s['osc'])
        k = sp[l3].argmax()
        sp = sp / sp[k]
        shift = x_l3 - g[k]
        name = tag or 'tPBE'
        print(f'  {name}: L3 maximum at {g[k]:.2f} eV, '
              f'shifted by {shift:+.2f} eV')
        off = top - OFFSET * (n + 1)
        ax.plot(g + shift, sp + off, color=col, lw=2,
                label=f'CASCI + {name}, shifted {shift:+.2f} eV')
        ax.text(x.max(), off + 0.06, name, color=INK, ha='right',
                va='bottom', fontsize=9)
    ax.axvline(x_l3, color=MUTED, lw=0.8, ls=':', zorder=0)
    ax.set_xlim(x.min(), x.max())
    ax.set_yticks([])
    ax.set_xlabel('Photon energy (eV)', color=INK)
    ax.set_ylabel('Intensity (normalized to L$_3$ maximum, offset)',
                  color=INK)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_title(f'{cfg.get("name", "")} {METAL} L-edge XAS: '
                 'CASCI/MC-PDFT + SOC vs experiment'.strip()
                 + f'\nLorentzian half width '
                   f'{common.GAMMA_CORE_EV if gamma is None else gamma:g} eV',
                 color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    out = os.path.join(RUN_DIR, 'xas_vs_experiment'
                       + ('' if gamma is None else f'_g{gamma:g}') + '.png')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f'Saved {os.path.relpath(out)}')
