import numpy as np

from graph_portraits import er_portraits, rrg_portraits, ba_portraits, ws_portraits
from plot_utils import portrait_figs
from figlist_to_mkv import create_mkv


def er(N=100, p_start=0.001, p_end=0.101, p_step=0.001, fps=5, n_ensemble=100, color_log_scale=True, color_abs_scale=True):
    p_range = np.arange(p_start, p_end, p_step)
    gen_ensemble = not((n_ensemble is None) or (n_ensemble <= 1))
    portraits, labels = er_portraits(n=N, p_range=p_range, n_ensemble=n_ensemble, p_precision=4)
    figlist = portrait_figs(portraits=portraits, labels=labels, same_shape=True,
                            color_log_scale=color_log_scale, color_abs_scale=color_abs_scale)
    path = f"./media/"
    name = "er_" + gen_ensemble * f"ensemble{n_ensemble}_" + f"n{N}_p{p_start}_{p_end}_fps{fps}"
    create_mkv(figlist, path, name, fps=fps)


def rrg(N=500, d_start=3, d_end=31, fps=3, n_ensemble=100, color_log_scale=True, color_abs_scale=True):
    d_range = np.arange(d_start, d_end)
    gen_ensemble = not((n_ensemble is None) or (n_ensemble <= 1))
    portraits, labels = rrg_portraits(n=N, d_range=d_range, n_ensemble=n_ensemble)
    figlist = portrait_figs(portraits=portraits, labels=labels, same_shape=True,
                            color_log_scale=color_log_scale, color_abs_scale=color_abs_scale)
    path = f"./media/"
    name = "rrg_" + gen_ensemble * f"ensemble{n_ensemble}_" + f"n{N}_d{d_start}_{d_end}_fps{fps}"
    create_mkv(figlist, path, name, fps=fps)


def ba(N=500, m_start=3, m_end=31, fps=3, n_ensemble=None, color_log_scale=True, color_abs_scale=True):
    m_range = np.arange(m_start, m_end)
    gen_ensemble = not((n_ensemble is None) or (n_ensemble <= 1))
    portraits, labels = ba_portraits(n=N, m_range=m_range, n_ensemble=n_ensemble)
    figlist = portrait_figs(portraits=portraits, labels=labels, same_shape=True,
                            color_log_scale=color_log_scale, color_abs_scale=color_abs_scale)
    path = f"./media/"
    name = "ba_" + gen_ensemble * f"ensemble{n_ensemble}_" + f"n{N}_m{m_start}_{m_end}_fps{fps}"
    create_mkv(figlist, path, name, fps=fps)


def ws(N=500, k=5, p_start=0.001, p_end=0.100, p_step=0.001, fps=5, n_ensemble=None, color_log_scale=True, color_abs_scale=True):
    p_range = np.arange(p_start, p_end, p_step)
    gen_ensemble = not((n_ensemble is None) or (n_ensemble <= 1))
    portraits, labels = ws_portraits(n=N, k=k, p_range=p_range, n_ensemble=n_ensemble, p_precision=4)
    figlist = portrait_figs(portraits=portraits, labels=labels, same_shape=True,
                            color_log_scale=color_log_scale, color_abs_scale=color_abs_scale)
    path = f"./media/"
    name = "ws_" + gen_ensemble * f"ensemble{n_ensemble}_" + f"n{N}_k{k}_p{p_start}_{p_end}_fps{fps}"
    create_mkv(figlist, path, name, fps=fps)
