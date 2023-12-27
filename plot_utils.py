from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt

from portrait_utils import to_same_shape

# set mpl rcParams
mpl.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
mpl.rcParams['font.family'] = 'sans-serif'

# TeX
# mpl.rc('text.latex', preamble=r'\\usepackage[russian]{babel}')
# mpl.rcParams['text.usetex'] = True

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.major.size'] = 14
mpl.rcParams['ytick.major.size'] = 14
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

mpl.rcParams["errorbar.capsize"] = 3

# mpl.rcParams['axes.edgecolor'] = "white"
# mpl.rcParams['xtick.color'] = "white"
# mpl.rcParams['ytick.color'] = "white"

# mpl.rcParams['xtick.labelcolor'] = "black"
# mpl.rcParams['ytick.labelcolor'] = "black"

mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['axes.labelsize'] = 30


def portrait_fig(port, color_log_scale=True, figsize=(40, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("distance l")
    ax.set_xlabel("neighbours k")
    ax.invert_yaxis()

    port_ = port + 1
    vmin = port_.min()
    vmax = port_.max()
    if color_log_scale:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(port_, norm=norm)
    fig.colorbar(im, ax=ax)
    return fig, ax


def portrait_figs(portraits, labels, same_shape=True, color_log_scale=True, color_abs_scale=True):
    if same_shape:
        portraits = to_same_shape(portraits)

    figs, axes = [], []

    # if color_abs_scale
    vmin = min([p.min() for p in portraits]) + 1
    vmax = max([p.max() for p in portraits]) + 1

    pbar = tqdm(portraits)
    for i, port in enumerate(pbar):
        pbar.set_description("Portrait figs")
        fig, ax = plt.subplots(figsize=(40, 12))
        ax.set_title(labels[i])
        ax.set_ylabel("distance l")
        ax.set_xlabel("neighbours k")
        ax.invert_yaxis()

        port_ = port + 1
        if not color_abs_scale:
            vmin = port_.min()
            vmax = port_.max()
        if color_log_scale:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(port_, norm=norm)
        fig.colorbar(im, ax=ax)

        figs.append(fig)
        axes.append(ax)
        plt.close(fig)

    return figs
