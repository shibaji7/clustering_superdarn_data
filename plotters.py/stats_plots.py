import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, num2date
from matplotlib import patches
import matplotlib.patches as mpatches
from matplotlib import ticker, cm, colors

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../utilities/")

def plot_acfs(rad="kap", fname="figs/acfs.png"):
    w_range, v_range = np.arange(0, 1001, 5), np.arange(0, 1001, 5)
    w, v = np.meshgrid(w_range, v_range) 
    cmap = cm.gray
    cmap.set_bad(color="k")
    levs = [10**c for c in np.linspace(-6, 0, 10, dtype=float)]
    fig = plt.figure(figsize=(5, 4), dpi=100)
    X = np.load("../../SuperDARN-Clustering/sd/data/%s.acfs.npy"%rad)
    count = np.nansum(X)
    X = X / count
    ax = fig.add_subplot(111)
    cs = ax.contour(w, v, X, levs, linewidths=0.5, colors='k', norm=colors.LogNorm())
    ax.clabel(cs, levels=levs, inline=1, fontsize=6, fmt=matplotlib.ticker.LogFormatterSciNotation())
    cntr = ax.contourf(w, v, X, levs, norm=colors.LogNorm(), cmap=cmap)
    ax.set_xlim(5, 100)
    ax.set_ylim(5, 100)
    cb = fig.colorbar(cntr, ax=ax, shrink=0.7)
    cb.set_label(r"$P(w,v), s^{2}m^{-2}$")
    ax.set_xlabel(r"Spectral Width (W), $ms^{-1}$")
    ax.set_ylabel(r"Velocity (V), $ms^{-1}$")
    ax.plot(w_range, equations[0](w_range), ls="--", color="r", lw=1., label=r"$|v|+\frac{w}{3}\leq 30$")
    ax.plot(w_range, equations[1](w_range), ls="--", color="b", lw=1., label=r"$|v|+\frac{w}{4}\leq 60$")
    ax.plot(w_range, equations[2](w_range), ls="--", color="g", lw=1., label=r"$|v|-0.139w+0.00113w^2\leq 33.1$")
    #ax.legend(loc=1)
    ax.text(0.25, 1.05, "Rad:"+rad +"(2011-2015)", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
    ax.text(0.75, 1.05, r"ACFs~%.2f$\times 10^6$"%(count/1e6), horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
    fig.savefig("iacf.png", bbox_inches="tight")
    return