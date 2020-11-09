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
from plot_utils import *


equations = {
        0: lambda w: 30 - w/3,
        1: lambda w: -w*0.25+60,
        2: lambda w: (0.139*w)-(0.00113*w**2)+33.1,
        3: lambda v, a, b, c: c - (a*(v-b)**2),
        }

def plot_lims(f=False):
    w = np.arange(0, 101)
    v = np.arange(0, 101)
    cmap = cm.PuBu_r
    cmap.set_bad(color="k")
    fig = plt.figure(figsize=(4,4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"Spectral Width (W), $ms^{-1}$")
    ax.set_ylabel(r"Velocity (V), $ms^{-1}$")
    ax.plot(w, equations[0](w), ls="--", color="r", lw=1., label=r"$|v|+\frac{w}{3}\leq 30$")
    ax.plot(w, equations[1](w), ls="--", color="b", lw=1., label=r"$|v|+\frac{w}{4}\leq 60$")
    ax.plot(w, equations[2](w), ls="--", color="g", lw=1., label=r"$|v|-0.139w+0.00113w^2\leq 33.1$")
    if f: ax.plot(equations[3](v, 0.1, 0, 50), v, ls="-", color="k", lw=1., label=r"$w-a\times (v-b)^2\leq c$")
    ax.legend(loc=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    if f: fig.savefig("iapdf.png", bbox_inches="tight")
    else: fig.savefig("ipdf.png", bbox_inches="tight")
    return

def plot_acfs(rad="kap"):
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

def plot_rad_acfs():
    rads = ["bks", "kap", "inv"]
    w_range, v_range = np.arange(0, 1001, 5), np.arange(0, 1001, 5)
    w, v = np.meshgrid(w_range, v_range)
    cmap = cm.gray
    cmap.set_bad(color="k")
    levs = [10**c for c in np.arange(-6, 0, dtype=float)]
    fig, axes = plt.subplots(figsize=(17, 4), dpi=100, nrows=1, ncols=3)
    for i, rad in enumerate(rads):
        X = np.load("../../SuperDARN-Clustering/sd/data/%s.acfs.npy"%rad)
        count = np.nansum(X)
        X = X / count
        ax = axes[i]
        cs = ax.contour(w, v, X, levs, linewidths=0.5, colors='k', norm=colors.LogNorm())
        ax.clabel(cs, levels=levs, inline=1, fontsize=6, fmt=matplotlib.ticker.LogFormatterSciNotation())
        cntr = ax.contourf(w, v, X, levs, norm=colors.LogNorm(), cmap=cmap)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        if i==2:
            cb = fig.colorbar(cntr, ax=axes.tolist(), shrink=0.7)
            cb.set_label(r"$P(w,v), s^{2}m^{-2}$")
        ax.set_xlabel(r"Spectral Width (W), $ms^{-1}$")
        if i ==0: ax.set_ylabel(r"Velocity (V), $ms^{-1}$")
        ax.text(0.25, 1.05, "Rad:"+rad +"(2011-2015)", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
        ax.text(0.75, 1.05, r"ACFs~%.2f$\times 10^6$"%(count/1e6), horizontalalignment="center", 
                verticalalignment="center", transform=ax.transAxes)
    fig.savefig("acf.png", bbox_inches="tight")
    return

def plot_hist_hr():
    rads = ["sas", "cvw"]
    fig = plt.figure(figsize=(8, 3), dpi=130)
    import pickle
    import glob
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    colors = ["r", "g", "b", "m", "orange"]
    for i, rad in enumerate(rads):
        ax = fig.add_subplot(121+i)    
        files = glob.glob("../data/%s*.pickle"%rad)
        v = []
        print(files)
        for f in files:
            data_dict = pickle.load(open(f, 'rb'))
            for vx in data_dict["vel"]:
                v.extend(vx)
        gm = GaussianMixture(5)
        ax.set_xlabel(r"Velocity (V), $ms^{-1}$")
        if i ==0: ax.set_ylabel("Density")
        v = np.sign(v)*np.log10(np.abs(v))
        gm.fit(np.reshape(v, (len(v),1)))
        xs = np.linspace(-4,4,1001)
        ax.hist(v, bins=2*301, histtype="step", lw=0.5, color="r", density=True)
        ax.set_xlim(-4,4)
        ax.set_ylim(0,0.4)
        #ax = ax.twinx()
        for z, m, vr in zip(range(5), gm.means_[:,0], gm.covariances_[:,0,0]):
            #ax.plot(xs, stats.norm.pdf(xs, loc=m, scale=np.sqrt(vr)), color=colors[z], ls="--", lw=0.4, alpha=0.7)
            pass
        ax.set_xticklabels([r"-$10^4$", r"-$10^2$", r"$10^0$", r"$10^2$", r"$10^4$"])
        ax.set_xlim(-4,4)
        #ax.set_ylim(0,1.)
        ax.text(0.25, 1.05, "Rad:"+rad +"(2011-2015)", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
        #ax.text(0.75, 1.05, r"ACFs~%.2f$\times 10^6$"%(count/1e6), horizontalalignment="center",
        #        verticalalignment="center", transform=ax.transAxes)
    fig.savefig("hist.png", bbox_inches="tight")
    return
