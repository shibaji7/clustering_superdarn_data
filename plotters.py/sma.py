import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN
import numpy as np
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
#import spacepy.plot as splot
#import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
import datetime as dt

#splot.style("spacepy_altgrid")
font = {"family": "serif", "color":  "black", "weight": "normal", "size": 10}
fonttext = {"family": "serif", "color":  "blue", "weight": "normal", "size": 10}

from matplotlib import font_manager
ticks_font = font_manager.FontProperties(family="serif", size=10, weight="normal")
matplotlib.rcParams["xtick.color"] = "k"
matplotlib.rcParams["ytick.color"] = "k"
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["mathtext.default"] = "default"


def smooth(x, window_len=51, window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def to_midlatitude_gate_summary(rad, df, gate_lims, names, smooth, fname, sb):
    """ Plot gate distribution summary """
    fig, axes = plt.subplots(figsize=(5,5), nrows=4, ncols=1, sharey="row", sharex=True, dpi=150)
    attrs = ["p_l"]
    beams = sb[1]
    scans = sb[0]
    count = sb[2]
    xpt = 100./scans
    labels = ["Power (dB)"]
    for j, attr, lab in zip(range(1), attrs, labels):
        ax = axes[j]
        ax.scatter(rand_jitter(df.slist), rand_jitter(df[attr]), color="r", s=1)
        ax.grid(color="gray", linestyle="--", linewidth=0.3)
        ax.set_ylabel(lab, fontdict=font)
        ax.set_xlim(0,110)
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax = axes[-3]
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.scatter(df.groupby(["slist"]).count().reset_index()["slist"], xpt*df.groupby(["slist"]).count().reset_index()["p_l"], color="k", s=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.set_ylabel("%Count", fontdict=font)
    ax.set_xlim(0,110)
    fonttext["color"] = "k"
    ax = axes[-2]
    ax.scatter(smooth[0], xpt*smooth[1], color="k", s=3)
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlim(0,110)
    ax.set_ylabel("<%Count>", fontdict=font)
    ax = axes[-1]
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(color="gray", linestyle="--", linewidth=0.3)
    ax.set_xlim(0,110)
    ax.set_xlabel("Gate", fontdict=font)
    ax.set_ylabel(r"$d_{<\%Count>}$", fontdict=font)
    ds = pd.DataFrame()
    ds["x"], ds["y"] = smooth[0], smooth[1]
    for k in gate_lims.keys():
        l, u = gate_lims[k][0], gate_lims[k][1]
        du = ds[(ds.x>=l) & (ds.x<=u)]
        p = np.append(np.diff(du.y), [0])
        p[np.argmax(du.y)] = 0
        ax.scatter(du.x, xpt*p, color="k", s=3)
    ax.axhline(0, color="b", lw=0.4, ls="--")
    ax.scatter(smooth[0], smooth[1], color="r", s=1, alpha=.6)
    ax.scatter(ds.x, ds.y, color="b", s=0.6, alpha=.5)
    fonttext["size"] = 8
    for k, n in zip(gate_lims.keys(), names.keys()):
        if k >= 0:
            for j in range(len(axes)):
                ax = axes[j]
                ax.axvline(x=gate_lims[k][0], color="b", lw=0.6, ls="--")
                ax.axvline(x=gate_lims[k][1], color="darkgreen", lw=0.6, ls=":")
                #ax.text(np.mean(gate_lims[k])/110, 0.7, names[n],
                #    horizontalalignment="center", verticalalignment="center",
                #    transform=ax.transAxes, fontdict=fonttext)
    fig.suptitle("Rad-%s, %s [%s-%s] UT"%(rad, df.time.tolist()[0].strftime("%Y-%m-%d"), 
        df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M")) + "\n" + 
        r"$Beams=%s, N_{sounds}=%d, N_{gates}=%d$"%(beams, scans, 110), size=12)
    fig.savefig(fname, bbox_inches="tight")
    plt.close()
    return

def beam_gate_boundary_plots(boundaries, clusters, clust_idf, glim, blim, title, fname):
    """ Beam gate boundary plots showing the distribution of clusters """
    fig, ax = plt.subplots(figsize=(6,4), nrows=1, ncols=1, dpi=240)
    ax.set_ylabel("Gates", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xlim(blim[0]-1, blim[1] + 2)
    ax.set_ylim(glim[0], glim[1])
    for b in range(blim[0], blim[1] + 1):
        ax.axvline(b, lw=0.3, color="gray", ls="--")
        boundary = boundaries[b]
        for bnd in boundary:
            ax.plot([b, b+1], [bnd["lb"], bnd["lb"]], ls="--", color="b", lw=0.5)
            ax.plot([b, b+1], [bnd["ub"], bnd["ub"]], ls="--", color="g", lw=0.5)
            #ax.scatter([b+0.5], [bnd["peak"]], marker="*", color="k", s=3)
    for x in clusters.keys():
        C = clusters[x]
        for _c in C: 
            if clust_idf is None: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, "%02d"%int(x),
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
            else: ax.text(_c["bmnum"]+(1./3.), (_c["ub"]+_c["lb"])/2, clust_idf[x],
                    horizontalalignment="center", verticalalignment="center",fontdict=fonttext)
    ax.axvline(b+1, lw=0.3, color="gray", ls="--")
    ax.set_title(title)
    ax.set_xticks(np.arange(blim[0], blim[1] + 1) + 0.5)
    ax.set_xticklabels(np.arange(blim[0], blim[1] + 1))
    fig.savefig(fname, bbox_inches="tight")
    return

def cluster_stats(df, cluster, fname, title):
    fig, axes = plt.subplots(figsize=(4,8), nrows=3, ncols=1, dpi=120, sharey=True)
    v, w, p = [], [], []
    for c in cluster:
        v.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].v.tolist())
        w.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].w_l.tolist())
        p.extend(df[(df.bmnum==c["bmnum"]) & (df.slist>=c["lb"]) & (df.slist>=c["lb"])].p_l.tolist())
    ax = axes[0]
    v, w, p = np.array(v), np.array(w), np.array(p)
    v[v<-1000] = -1000
    v[v>1000] = 1000
    l, u = np.quantile(v,0.1), np.quantile(v,0.9)
    ax.axvline(np.sign(l)*np.log10(abs(l)), ls="--", lw=1., color="r")
    ax.axvline(np.sign(u)*np.log10(abs(u)), ls="--", lw=1., color="r")
    ax.hist(np.sign(v)*np.log10(abs(v)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"V, $ms^{-1}$")
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])
    ax.text(0.7, 0.7, r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(v[(v>l) & (v<u)]), np.median(v[(v>l) & (v<u)])) + "\n"\
            + r"$V_{max}$=%.1f, $V_{min}$=%.1f"%(np.max(v[(v>l) & (v<u)]), np.min(v[(v>l) & (v<u)])) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(v[(v>l) & (v<u)]),len(v[(v>l) & (v<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax = axes[1]
    w[w>100]=100
    w[w<-100] = 100
    l, u = np.quantile(w,0.1), np.quantile(w,0.9)
    ax.axvline(l, ls="--", lw=1., color="r")
    ax.axvline(u, ls="--", lw=1., color="r")
    ax.hist(w, bins=range(-100,100,1), histtype="step", density=False)
    ax.set_xlabel(r"W, $ms^{-1}$")
    ax.set_xlim([-100,100])
    ax.set_ylabel("Counts")
    ax.text(0.75, 0.8, r"$W_{\mu}$=%.1f, $\hat{W}$=%.1f"%(np.mean(w[(w>l) & (w<u)]), np.median(w[(w>l) & (w<u)])) + "\n"\
            + r"$W_{max}$=%.1f, $W_{min}$=%.1f"%(np.max(w[(w>l) & (w<u)]), np.min(w[(w>l) & (w<u)])) + "\n"\
            + r"$W_{\sigma}$=%.1f, n=%d"%(np.std(w[(w>l) & (w<u)]),len(w[(w>l) & (w<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax = axes[2]
    p[p>30]=30
    l, u = np.quantile(p,0.1), np.quantile(p,0.9)
    ax.axvline(l, ls="--", lw=1., color="r")
    ax.axvline(u, ls="--", lw=1., color="r")
    ax.hist(p, bins=range(0,30,1), histtype="step", density=False)
    ax.set_xlabel(r"P, $dB$")
    ax.set_xlim([0,30])
    ax.set_ylabel("Counts")
    ax.text(0.75, 0.8, r"$P_{\mu}$=%.1f, $\hat{P}$=%.1f"%(np.mean(p[(p>l) & (p<u)]), np.median(p[(p>l) & (p<u)])) + "\n"\
            + r"$P_{max}$=%.1f, $P_{min}$=%.1f"%(np.max(p[(p>l) & (p<u)]), np.min(p[(p>l) & (p<u)])) + "\n"\
            + r"$P_{\sigma}$=%.1f, n=%d"%(np.std(p[(p>l) & (p<u)]),len(p[(p>l) & (p<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.5)
    fig.savefig(fname, bbox_inches="tight")
    return

def general_stats(g_stats, fname):
    fig, axes = plt.subplots(figsize=(6,5), nrows=2, ncols=1, dpi=120, sharex=True)
    ax = axes[0]
    width=0.2
    df = pd.DataFrame.from_records(list(g_stats.values()))
    ax.bar(df.bmnum-width, df.sound, width=0.3, color="r", label="S")
    ax.set_ylabel(r"$N_{sounds}$", fontdict=font)
    ax.legend(loc=2)
    ax = ax.twinx()
    ax.bar(df.bmnum+width, df.echo, width=0.3, color="b", label="E")
    ax.set_ylabel(r"$N_{echo}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax.legend(loc=1)

    ax = axes[1]
    ax.errorbar(df.bmnum, df.v, yerr=df.v_mad, color="r", elinewidth=2.5, ecolor="r", fmt="o", ls="None", label="V")
    ax.errorbar(df.bmnum, df.w, yerr=df.w_mad, color="b", elinewidth=1.5, ecolor="b", fmt="o", ls="None", label="W")
    ax.errorbar(df.bmnum, df.p, yerr=df.p_mad, color="k", elinewidth=0.5, ecolor="k", fmt="o", ls="None", label="P")
    ax.set_ylim(-20, 40)
    ax.set_ylabel(r"$V_{med},W_{med},P_{med}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(df.bmnum)
    ax.legend(loc=1)

    fig.subplots_adjust(hspace=0.1)
    fig.savefig(fname, bbox_inches="tight")
    return


def individal_cluster_stats(cluster, df, fname, title):
    fig = plt.figure(figsize=(8,12), dpi=120)
    V = []
    vbbox, vbox = {}, []
    vecho, vsound, echo, sound = {}, {}, [], []
    beams = []
    for c in cluster:
        v = np.array(df[(df.slist>=c["lb"]) & (df.slist<=c["ub"]) & (df.bmnum==c["bmnum"])].v)
        V.extend(v.tolist())
        v[v<-1000] = -1000
        v[v>1000] = 1000
        if c["bmnum"] not in vbbox.keys(): 
            vbbox[c["bmnum"]] = v.tolist()
            vecho[c["bmnum"]] = c["echo"]
            vsound[c["bmnum"]] = c["sound"]
        else: 
            vbbox[c["bmnum"]].extend(v.tolist())
            vecho[c["bmnum"]] += c["echo"]
            #vsound[c["bmnum"]] += c["sound"]
    beams = sorted(vbbox.keys())
    avbox = []
    for b in beams:
        if b!=15: avbox.append(vbbox[b])
        vbox.append(vbbox[b])
        echo.append(vecho[b])
        sound.append(vsound[b])
    from scipy import stats
    pval = -1.
    if len(vbox) > 1: 
        H, pval = stats.f_oneway(*avbox)
        print(H,pval)
    ax = plt.subplot2grid((4, 2), (0, 1), colspan=1)
    V = np.array(V)
    V[V<-1000] = -1000
    V[V>1000] = 1000
    l, u = np.quantile(V,0.05), np.quantile(V,0.95)
    ax.axvline(np.sign(l)*np.log10(abs(l)), ls="--", lw=1., color="r")
    ax.axvline(np.sign(u)*np.log10(abs(u)), ls="--", lw=1., color="r")
    ax.hist(np.sign(V)*np.log10(abs(V)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.text(0.7, 0.8, r"$V_{min}$=%.1f, $V_{max}$=%.1f"%(np.min(V[(V>l) & (V<u)]), np.max(V[(V>l) & (V<u)])) + "\n"\
            + r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(V[(V>l) & (V<u)]), np.median(V[(V>l) & (V<u)])) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(V[(V>l) & (V<u)]),len(V[(V>l) & (V<u)])),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_xlabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_yticklabels([])
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])

    ax = plt.subplot2grid((4, 2), (0, 0), colspan=1)
    ax.hist(np.sign(V)*np.log10(abs(V)), bins=np.linspace(-3,3,101), histtype="step", density=False)
    ax.text(0.7, 0.8, r"$V_{min}$=%.1f, $V_{max}$=%.1f"%(np.min(V), np.max(V)) + "\n"\
            + r"$V_{\mu}$=%.1f, $\hat{V}$=%.1f"%(np.mean(V), np.median(V)) + "\n"\
            + r"$V_{\sigma}$=%.1f, n=%d"%(np.std(V),len(V)),
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_ylabel("Counts", fontdict=font)
    ax.set_xlabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_xlim([-3,3])
    ax.set_xticklabels([-1000,-100,-10,1,10,100,1000])
    
    ax = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    ax.boxplot(vbox, flierprops = dict(marker="o", markerfacecolor="none", markersize=0.8, linestyle="none"))
    ax.set_ylim(-100,100)
    ax.set_xlabel(r"Beams", fontdict=font)
    ax.set_ylabel(r"V, $ms^{-1}$", fontdict=font)
    ax.set_xticklabels(beams)
    ax.text(1.05,0.5, "p-val=%.2f"%pval, horizontalalignment="center", verticalalignment="center",
            transform=ax.transAxes, fontdict={"size":10}, rotation=90)
    ax.axhline(0, ls="--", lw=0.5, color="k")
    fig.suptitle(title, y=0.92)

    ax = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax.boxplot(vbox, flierprops = dict(marker="o", markerfacecolor="none", markersize=0.8, linestyle="none"),
            showbox=False, showcaps=False)
    ax.set_xticklabels(beams)
    ax.axhline(0, ls="--", lw=0.5, color="k")
    ax.set_xlabel(r"Beams", fontdict=font)
    ax.set_ylabel(r"V, $ms^{-1}$", fontdict=font)

    ax = plt.subplot2grid((4, 2), (3, 0), colspan=2)
    width=0.2
    ax.bar(np.array(beams)-width, sound, width=0.3, color="r", label="S")
    ax.set_ylabel(r"$N_{sounds}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.legend(loc=2)
    ax = ax.twinx()
    ax.bar(np.array(beams)+width, echo, width=0.3, color="b", label="E")
    ax.set_ylabel(r"$N_{echo}$", fontdict=font)
    ax.set_xlabel("Beams", fontdict=font)
    ax.set_xticks(beams)
    ax.legend(loc=1)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.savefig(fname, bbox_inches="tight")
    return



class MiddleLatFilter(object):
    """ Class to filter middle latitude radars """

    def __init__(self, rad, scans, eps=2, min_samples=10, plot=False):
        """
        initialize variables
        
        rad: Radar code
        scans: List of scans
        eps: Radius of DBSCAN
        min_samples: min samplese of DBSCAN
        """
        self.rad = rad
        self.scans = scans
        self.eps = eps
        self.min_samples = min_samples
        self.boundaries = {}
        self.plot = plot
        return
    
    def _reset_(self, rad, scans, plot=False):
        """ Reset existing parameters """
        self.rad = rad
        self.scans = scans
        self.plot = plot
        return

    def extract_gatelims(self, df):
        """
        Extract gate limits for individual clusters
        """
        glims = {}
        for l in set(df.labels):
            if l >= 0:
                u = df[df.labels==l]
                if len(u) > 0: glims[l] = [np.min(u.slist) + 1, np.max(u.slist) - 1]
        return glims

    def filter_by_dbscan(self, df, bm):
        """
        Do filter by dbscan name
        """
        du, sb = df[df.bmnum==bm], [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l), 
                "sound":sb[0], "echo":len(du), "v_mad": st.median_absolute_deviation(du.v), 
                "p_mad": st.median_absolute_deviation(du.p_l), "w_mad": st.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all":
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else:
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        print(" Beam Analysis: ", bm, len(du.groupby("time")))
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        if len(eco) > self.window: eco, Eco = smooth(eco, self.window), smooth(Eco, self.window)
        glims, labels = {}, []
        ds = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(du[["slist"]].values)
        du["labels"] = ds.labels_
        names = {}
        print(eco, Eco, bm)
        for j, r in enumerate(set(ds.labels_)):
            x = du[du.labels==r]
            glims[r] = [np.min(x.slist), np.max(x.slist)]
            names[r] = "C"+str(j)
            if r >= 0: self.boundaries[bm].append({"peak": Rng[np.min(x.slist) + Eco[np.min(x.slist):np.max(x.slist)].argmax()],
                "ub": np.max(x.slist), "lb": np.min(x.slist), "value": np.max(eco)*xpt, "bmnum": bm, "echo": len(x), "sound": sb[0]})
        print(" Individual clster detected: ", set(du.labels))
        if self.plot: to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco),
                "figs/{r}_gate_{bm}_summary.png".format(r=self.rad, bm="%02d"%bm), sb)
        return

    def filter_by_SMF(self, df, bm, method=None):
        """
        Filter by Simple Minded Filter
        method: np/ndimage
        """
        def local_minima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True): 
            """Find all local maxima of the array, separated by at least min_distance."""
            array = np.asarray(array)
            cval = 0 
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest" 
            else: mode = "constant" 
            cval = array.max()+1 
            min_points = array == ndimage.minimum_filter(array, 1+2*min_distance, mode=mode, cval=cval) 
            troughs = [indices[min_points] for indices in np.indices(array.shape)][0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] <= np.max(du.slist) - min_distance: troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        def local_maxima_ndimage(array, min_distance = 1, periodic=False, edges_allowed=True):
            array = np.asarray(array)
            cval = 0
            if periodic: mode = "wrap"
            elif edges_allowed: mode = "nearest"
            else: mode = "constant"
            cval = array.max()+1
            max_points = array == ndimage.maximum_filter(array, 1+2*min_distance, mode=mode, cval=cval)
            peaks = [indices[max_points] for indices in np.indices(array.shape)][0]
            return peaks

        def local_minima_np(array):
            """ Local minima by numpy stats """
            troughs = signal.argrelmin(eco, order=self.order)[0]
            if troughs[0] != 0: troughs = np.insert(troughs, 0, 0)
            if troughs[-1] != np.max(du.slist): troughs = np.append(troughs, [np.max(du.slist)])
            troughs[-2] = troughs[-2] + 1
            return troughs

        if bm == "all": du = df.copy()
        else: du = df[df.bmnum==bm]
        print(" Beam Analysis: ", bm, len(du.groupby("time")))
        sb = [np.nan, np.nan, np.nan]
        sb[0] = len(du.groupby("time"))
        self.gen_summary[bm] = {"bmnum": bm, "v": np.median(du.v), "p": np.median(du.p_l), "w": np.median(du.w_l),
                "sound":sb[0], "echo":len(du), "v_mad": st.median_absolute_deviation(du.v),
                "p_mad": st.median_absolute_deviation(du.p_l), "w_mad": st.median_absolute_deviation(du.w_l)}
        xpt = 100./sb[0]
        if bm == "all": 
            sb[1] = "[" + str(int(np.min(df.bmnum))) + "-" + str(int(np.max(df.bmnum))) + "]"
            sb[2] = len(self.scans) * int((np.max(df.bmnum)-np.min(df.bmnum)+1))
        else: 
            sb[1] = "[" + str(int(bm)) + "]"
            sb[2] = len(self.scans)
        rng, eco = np.array(du.groupby(["slist"]).count().reset_index()["slist"]),\
                np.array(du.groupby(["slist"]).count().reset_index()["p_l"])
        glims, labels = {}, []
        Rng, Eco = np.arange(np.max(du.slist)+1), np.zeros((np.max(du.slist)+1))
        for e, r in zip(eco, rng):
            Eco[Rng.tolist().index(r)] = e
        eco, Eco = smooth(eco, self.window), smooth(Eco, self.window)
        du["labels"] = [np.nan] * len(du)
        names = {}
        if method == "np": troughs = local_minima_np(np.array(eco))
        else: troughs = local_minima_ndimage(np.array(eco), min_distance=5)
        peaks = local_maxima_ndimage(np.array(eco), min_distance=5)
        print(" Gate bounds: ", troughs, peaks)
        peaks = np.append(peaks, np.median(troughs[-2:]))
        if len(troughs) > len(peaks): 
            troughs = troughs[:len(peaks)]
        if len(peaks) > len(troughs): peaks = peaks[:len(troughs)]
        for r in range(len(troughs)-1):
            glims[r] = [troughs[r], troughs[r+1]]
            du["labels"] = np.where((du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r]), r, du["labels"])
            names[r] = "C" + str(r)
            if r >= 0: self.boundaries[bm].append({"peak": peaks[r],
                "ub": troughs[r+1], "lb": troughs[r], "value": np.max(eco)*xpt, "bmnum": bm, 
                "echo": len(du[(du["slist"]<=troughs[r+1]) & (du["slist"]>=troughs[r])]), "sound": sb[0]})
        du["labels"] =  np.where(np.isnan(du["labels"]), -1, du["labels"])
        print(" Individual clster detected: ", set(du.labels))
        if self.plot: to_midlatitude_gate_summary(self.rad, du, glims, names, (rng,eco),
                "figs/{r}_gate_{bm}_summary.png".format(r=self.rad, bm="%02d"%bm), sb)
        return

    def doFilter(self, io, window=11, order=1, beams=range(4,24), dbeam=15):
        """
        Do filter for sub-auroral scatter
        """
        self.gen_summary = {}
        self.clust_idf = {}
        self.order = order
        self.window = window
        df = pd.DataFrame()
        for i, fsc in enumerate(self.scans):
            dx = io.convert_to_pandas(fsc.beams, v_params=["p_l", "v", "w_l", "slist"])
            df = df.append(dx)
        beams = range(np.min(df.bmnum), np.max(df.bmnum))
        print(" Beam range - ", np.min(df.bmnum), np.max(df.bmnum))
        if beams == None or len(beams) == 0: 
            bm = "all"
            self.boundaries[bm] = []
            self.filter_by_SMF(df, bm)
        else:
            for bm in beams:
                self.boundaries[bm] = []
                self.gen_summary[bm] = {}
                if (dbeam is not None) and (bm==dbeam): self.filter_by_SMF(df, bm)
                else: self.filter_by_dbscan(df, bm)
        title = "Date: %s [%s-%s] UT | %s"%(df.time.tolist()[0].strftime("%Y-%m-%d"),
                df.time.tolist()[0].strftime("%H.%M"), df.time.tolist()[-1].strftime("%H.%M"), self.rad.upper())
        fname = "figs/%s_gate_boundary.png"%(self.rad)
        self.gc = []
        for x in self.boundaries.keys():
            for j, m in enumerate(self.boundaries[x]):
                self.gc.append(m)
        self.ubeam, self.lbeam = np.max(df.bmnum)-1, np.min(df.bmnum)
        self.sma_bgspace()
        df["labels"] = [np.nan] * len(df)
        for ck in self.clusters.keys():
            cluster = self.clusters[ck]
            for c in cluster:
                df["labels"] = np.where((df["bmnum"]==c["bmnum"]) & (df["slist"]<=c["ub"]) & (df["slist"]>=c["lb"]), ck, df["labels"])
        #self.cluster_identification(df)
        if self.plot: beam_gate_boundary_plots(self.boundaries, self.clusters, None, 
                glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
                fname=fname)
        #if self.plot: beam_gate_boundary_plots(self.boundaries, self.clusters, self.clust_idf,
        #        glim=(0, 100), blim=(np.min(df.bmnum), np.max(df.bmnum)), title=title,
        #        fname=fname.replace("gate_boundary", "gate_boundary_id"))
        #if self.plot: general_stats(self.gen_summary, fname="figs/%s_general_stats.png"%(self.rad))
        #for c in self.clusters.keys():
        #    cluster = self.clusters[c]
        #    fname = "figs/%s_clust_%02d_stats.png"%(self.rad, c)
        #    cluster_stats(df, cluster, fname, title+" | Cluster# %02d"%c)
        #    individal_cluster_stats(self.clusters[c], df, "figs/%s_ind_clust_%02d_stats.png"%(self.rad, c), 
        #            title+" | Cluster# %02d"%c+"\n"+"Cluster ID: _%s_"%self.clust_idf[c].upper())
        return df

    def cluster_identification(self, df, qntl=[0.05,0.95]):
        """ Idenitify the cluster based on Idenitification """
        for c in self.clusters.keys():
            V = []
            for cls in self.clusters[c]:
                v = np.array(df[(df.slist>=cls["lb"]) & (df.slist<=cls["ub"]) & (df.bmnum==cls["bmnum"])].v)
                V.extend(v.tolist())
            V = np.array(V)
            l, u = np.quantile(V,qntl[0]), np.quantile(V,qntl[1])
            Vmin, Vmax = np.min(V[(V>l) & (V<u)]), np.max(V[(V>l) & (V<u)])
            Vmean, Vmed = np.mean(V[(V>l) & (V<u)]), np.median(V[(V>l) & (V<u)])
            Vsig = np.std(V[(V>l) & (V<u)])
            self.clust_idf[c] = "us"
            if Vmin > -20 and Vmax < 20: self.clust_idf[c] = "gs"
            elif (Vmin > -50 and Vmax < 50) and (Vmed-Vsig < -20 or Vmed+Vsig > 20): self.clust_idf[c] = "sais"
        return

    def sma_bgspace(self):
        """
        Simple minded algorithm in B.G space
        """
        def range_comp(x, y, pcnt=0.7):
            _cx = False
            insc = set(x).intersection(y)
            if len(x) < len(y) and len(insc) >= len(x)*pcnt: _cx = True
            if len(x) > len(y) and len(insc) >= len(y)*pcnt: _cx = True
            return _cx
        def find_adjucent(lst, mxx):
            mxl = []
            for l in lst:
                if l["peak"] >= mxx["lb"] and l["peak"] <= mxx["ub"]: mxl.append(l)
                elif mxx["peak"] >= l["lb"] and mxx["peak"] <= l["ub"]: mxl.append(l)
                elif range_comp(range(l["lb"], l["ub"]+1), range(mxx["lb"], mxx["ub"]+1)): mxl.append(l)
            return mxl
        def nested_cluster_find(bm, mx, j, case=-1):
            if bm < self.lbeam and bm > self.ubeam: return
            else:
                if (case == -1 and bm >= self.lbeam) or (case == 1 and bm <= self.ubeam):
                    mxl = find_adjucent(self.boundaries[bm], mx)
                    for m in mxl:
                        if m in self.gc:
                            del self.gc[self.gc.index(m)]
                            self.clusters[j].append(m)
                            nested_cluster_find(m["bmnum"] + case, m, j, case)
                            nested_cluster_find(m["bmnum"] + (-1*case), m, j, (-1*case))
                return
        self.clusters = {}
        j = 0
        while len(self.gc) > 0:
            self.clusters[j] = []
            mx = max(self.gc, key=lambda x:x["value"])
            self.clusters[j].append(mx)
            if mx in self.gc: del self.gc[self.gc.index(mx)]
            nested_cluster_find(mx["bmnum"] - 1, mx, j, case=-1)
            nested_cluster_find(mx["bmnum"] + 1, mx, j, case=1)
            j += 1
        return
    
class ScatterTypeDetection(object):
    """ Detecting scatter type """

    def __init__(self, df):
        """ kind: 0- individual, 2- KDE by grouping """
        self.df = df
        return

    def run(self, kind=0, thresh=[1./3.,2./3.], case=0, mod=False):
        self.kind = kind
        self.thresh = thresh
        self.case = case
        if self.kind == 0: self.indp()
        if self.kind == 1: self.group()
        if self.kind == 2: self.kde()
        if mod: self.gs_flg[self.gs_flg==2] = 1
        self.df["gflg"] = self.gs_flg
        return self.df

    def kde(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["labels"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1: self.gs_flg[clust_mask] = -1
            else:
                v, w = vel[clust_mask], wid[clust_mask]
                if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                #if self.case == 3: f = 1/(1+np.exp(np.abs(w)-0.1*(v-0)**2-10))
                if self.case == 3: 
                    f = 1/(1+np.exp(np.abs(v)-20.))
                gflg = np.median(f)
                if gflg <= self.thresh[0]: gflg=0.
                elif gflg >= self.thresh[1]: gflg=1.
                else: gflg=-1
                self.gs_flg[clust_mask] = gflg
        return

    def indp(self):
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["labels"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1: self.gs_flg[clust_mask] = -1
            else:
                v, w = vel[clust_mask], wid[clust_mask]
                gflg = np.zeros(len(v))
                if self.case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
                if self.case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
                if self.case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
                #if self.case == 3: gflg = (np.abs(w)-0.1*(v-0)**2<10).astype(int)
                if self.case == 3: 
                    for i, vi, wi in zip(range(len(v)),v,w):
                        if np.abs(vi)<10: gflg[i] = 1
                        elif np.abs(vi)>=15 and np.abs(vi)<50: gflg[i] = 2
                        elif np.abs(vi)>=50: gflg[i] = 0
                    #gflg = (np.logical_or(np.abs(v)<20., np.abs(w)<30.)).astype(int)
                self.gs_flg[clust_mask] = gflg
        return

    def group(self, type="median"):
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["labels"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        for c in np.unique(clust_flg_1d):
            clust_mask = c == clust_flg_1d
            if c == -1: self.gs_flg[clust_mask] = -1
            else:
                v, w = np.mean(vel[clust_mask]), np.mean(wid[clust_mask])
                v, w = vel[clust_mask], wid[clust_mask]
                gflg = np.zeros(len(v))
                if self.case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
                if self.case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
                if self.case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
                #if self.case == 3: 
                #    vl, vu = np.quantile(vel[clust_mask],0.25), np.quantile(vel[clust_mask],0.75)
                #    wl, wu = np.quantile(vel[clust_mask],0.25), np.quantile(vel[clust_mask],0.75)
                #    v, w = vel[clust_mask], wid[clust_mask]
                #    v, w = v[(v>vl) & (v<vu)], w[(w>wl) & (w<wu)]
                #    gflg = -1
                #    #if ((vu < 10) and (vl > -10.)) and (wu < 25.): gflg = 1
                #    if np.mean(np.abs(v))<5: gflg=1
                #    elif np.mean(np.abs(v))>=5 and np.mean(np.abs(v))<20: gflg = 2
                #    elif np.mean(np.abs(v))>=20: gflg = 0
                #self.gs_flg[clust_mask] = gflg
                if self.case == 3: 
                    for i, vi, wi in zip(range(len(v)),v,w):
                        if np.abs(vi)<5: gflg[i] = 1
                        elif np.abs(vi)>=5 and np.abs(vi)<50: gflg[i] = 2
                        elif np.abs(vi)>=50: gflg[i] = 0
                    #gflg = (np.logical_or(np.abs(v)<20., np.abs(w)<30.)).astype(int)
                self.gs_flg[clust_mask] = max(set(gflg.tolist()), key = gflg.tolist().count) 
        return