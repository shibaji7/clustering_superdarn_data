import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, num2date
from matplotlib import patches
import matplotlib.patches as mpatches
import random
import pytz

import datetime as dt
import pandas as pd

import pydarn
from astral.sun import sun
from astral import Observer

import utils

LFS = "LFS/LFS_clustering_superdarn_data/"

CLUSTER_CMAP = plt.cm.gist_rainbow

def get_cluster_cmap(n_clusters, plot_noise=False):
    cmap = CLUSTER_CMAP
    cmaplist = [cmap(i) for i in range(cmap.N)]
    while len(cmaplist) < n_clusters:
        cmaplist.extend([cmap(i) for i in range(cmap.N)])
    cmaplist = np.array(cmaplist)
    r = np.array(range(len(cmaplist)))
    random.seed(10)
    random.shuffle(r)
    cmaplist = cmaplist[r]
    if plot_noise:
        cmaplist[0] = (0, 0, 0, 1.0)    # black for noise
    rand_cmap = cmap.from_list("Cluster cmap", cmaplist, len(cmaplist))
    return rand_cmap


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=100) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=15)
        mpl.rcParams.update({"font.size": 10})
        return
    
    def lay_sunrise_sunset(self, ax, ss_obj):
        gate = ss_obj["gate"]
        sunset = ss_obj["sunset"]
        sunrise = ss_obj["sunrise"]
        ax.plot(sunrise, gate, color="k", lw=1, ls="--")
        ax.plot(sunset, gate, color="k", lw=1, ls="--")
        return
        
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, p_step=25, xlabel="Time UT", zparam="v",
                    label="Velocity [m/s]", ax=None, fig=None, addcb=True, ss_obj=None):
        if ax is None: ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = plt.cm.jet
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        if fig is None: fig = self.fig
        if addcb: self._add_colorbar(fig, ax, bounds, cmap, label=label)
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        if ss_obj: self.lay_sunrise_sunset(ax, ss_obj)
        return
    
    def addCluster(self, df, beam, title, xlabel="", label_clusters=True, skill=None, ax=None, ss_obj=None):
        # add new axis
        if ax is None: ax = self._add_axis()
        df = df[df.bmnum==beam]
        unique_labs = np.sort(np.unique(df.labels))
        for i, j in zip(range(len(unique_labs)), unique_labs):
            if j > 0:
                df["labels"]=np.where(df["labels"]==j, i, df["labels"])
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam="labels")
        flags = df.labels
        if -1 in flags:
            cmap = get_cluster_cmap(len(np.unique(flags)), plot_noise=True)       # black for noise
        else:
            cmap = get_cluster_cmap(len(np.unique(flags)), plot_noise=False)

        # Lower bound for cmap is inclusive, upper bound is non-inclusive
        bounds = list(range( len(np.unique(flags)) ))    # need (max_cluster+1) to be the upper bound
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_title(title,  loc="left", fontdict={"fontweight": "bold"})
        if skill is not None:
            txt = r"CH = %.1f, BH = %.1f $\times 10^{6}$"%(skill.chscore, skill.bhscore/1e6) +"\n"+\
                    "H = %.1f, Xu = %.1f"%(skill.hscore, skill.xuscore)
            ax.text(0.8, 0.8, txt, horizontalalignment="center",
                    verticalalignment="center", transform=ax.transAxes)
        if label_clusters:
            num_flags = len(np.unique(flags))
            for f in np.unique(flags):
                flag_mask = Z.T==f
                g = Y[flag_mask].astype(int)
                t_c = X[flag_mask]
                # Only label clusters large enough to be distinguishable on RTI map,
                # OR label all clusters if there are few
                if (len(t_c) > 250 or
                   (num_flags < 50 and len(t_c) > 0)) \
                   and f != -1:
                    m = int(len(t_c) / 2)  # Time is sorted, so this is roughly the index of the median time
                    ax.text(t_c[m], g[m], str(int(f)), fontdict={"size": 8, "fontweight": "bold"})  # Label cluster #
        return
    
    def addGSIS(self, df, beam, title, xlabel="", zparam="gflg_0", clusters=None, label_clusters=False, ax=None, ss_obj=None):
        # add new axis
        if ax is None: ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam,)
        flags = np.array(df[zparam]).astype(int)
        if -1 in flags and 2 in flags:                     # contains noise flag
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                (1.0, 0.0, 0.0, 1.0),     # blue
                (0.0, 0.0, 1.0, 1.0),     # red
                (0.0, 1.0, 0.0, 1.0)])    # green
            bounds = [-1, 0, 1, 2, 3]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US"), mpatches.Patch(color="green", label="SAIS")]
        elif -1 in flags and 2 not in flags:
            cmap = mpl.colors.ListedColormap([(0.0, 0.0, 0.0, 1.0),     # black
                                              (1.0, 0.0, 0.0, 1.0),     # blue
                                              (0.0, 0.0, 1.0, 1.0)])    # red
            bounds = [-1, 0, 1, 2]      # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS"),
                      mpatches.Patch(color="black", label="US")]
        else:
            cmap = mpl.colors.ListedColormap([(1.0, 0.0, 0.0, 1.0),  # blue
                                              (0.0, 0.0, 1.0, 1.0)])  # red
            bounds = [0, 1, 2]          # Lower bound inclusive, upper bound non-inclusive
            handles = [mpatches.Patch(color="red", label="IS"), mpatches.Patch(color="blue", label="GS")]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_title(title,  loc="left", fontdict={"fontweight": "bold"})
        ax.legend(handles=handles, loc=4)
        if label_clusters:
            flags = df.labels
            num_flags = len(np.unique(flags))
            X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="slist", zparam="labels")
            for f in np.unique(flags):
                flag_mask = Z.T==f
                g = Y[flag_mask].astype(int)
                t_c = X[flag_mask]
                # Only label clusters large enough to be distinguishable on RTI map,
                # OR label all clusters if there are few
                if (len(t_c) > 250 or
                   (num_flags < 100 and len(t_c) > 0)) \
                   and f != -1:
                    tct = ""
                    m = int(len(t_c) / 2)  # Time is sorted, so this is roughly the index of the median time
                    if clusters[beam][int(f)]["type"] == "IS": tct = "%.1f IS"%((1-clusters[beam][int(f)]["auc"])*100)
                    if clusters[beam][int(f)]["type"] == "GS": tct = "%.1f GS"%(clusters[beam][int(f)]["auc"]*100)
                    ax.text(t_c[m], g[m], tct, fontdict={"size": 8, "fontweight": "bold", "color":"gold"})  # Label cluster #
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    # Private helper functions

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def _add_colorbar(self, fig, ax, bounds, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return
        
def summary_skill_scores(rads, dates, a_name, gmm=False, etype=""):
    if gmm: fname = "../outputs/skills/{rad}.{a_name}.gmm.{dn}.csv"
    else: fname = "../outputs/skills/{rad}.{a_name}.{dn}.csv"
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    X = pd.DataFrame()
    for rad, dn in zip(rads, dates):
        floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
        if utils.fetch_file(conn, floc, LFS):
            y = pd.read_csv(floc)
            X = pd.concat([X,y])
    X = X.median().to_frame().T
    X["model"] = a_name
    X["gmm"] = gmm
    X["rad"] = rad
    X["event_type"] = etype
    if not os.path.exists("../outputs/skill.csv"): X.to_csv("../outputs/skill.csv", index=False, float_format="%g")
    else:
        y = pd.read_csv("../outputs/skill.csv")
        y = pd.concat([y, X])
        y = y.drop_duplicates()
        y.to_csv("../outputs/skill.csv", index=False, float_format="%g")
    os.system("rm ../outputs/skills/*")
    return
        
def histograms_skill_scores(rads, a_name, starts, ends, gmm=False):
    if gmm: fname = "../outputs/skills/{rad}.{a_name}.{dn}.csv"
    else: fname = "../outputs/skills/{rad}.{a_name}.gmm.{dn}.csv"
    X = []
    for rad, start, end in zip(rads, starts, ends):
        dn = start
        while dn <= end:
            floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
            dn = dn + dt.timedelta(days=1)
            y = pd.read_csv(floc)
            X.append(y.values[0].tolist())
    X = np.array(X)
    fig, axes = plt.subplots(dpi=150, figsize=(6,6), sharey = True, nrows=2, ncols=2)
    ax = axes[0,0]
    ax.hist(X[:,0], histtype="step", lw=0.5, ls="--", color="r")
    ax.set_xlabel("CH Score")
    ax.set_ylabel("Counts")
    ax.text(0.2, 0.9, r"$\mu_{CH}=$"+"%.2f"%np.mean(X[:,0]), ha="left", va="center", transform=ax.transAxes)
    ax = axes[0,1]
    ax.hist(X[:,1], histtype="step", lw=0.5, ls="--", color="r")
    ax.set_xscale("log")
    ax.set_xlabel("BH Index")
    ax.text(0.2, 0.9, r"$\mu_{BH}=$"+"%.2f"%(np.mean(X[:,1])/1e7)+r"$\times 10^{7}$", ha="left", va="center", transform=ax.transAxes)
    ax = axes[1,0]
    ax.hist(X[:,2], histtype="step", lw=0.5, ls="--", color="r")
    ax.set_xlabel("H Score")
    ax.set_ylabel("Counts")
    ax.text(0.2, 0.9, r"$\mu_{H}=$"+"%.2f"%np.mean(X[:,2]), ha="left", va="center", transform=ax.transAxes)
    ax = axes[1,1]
    ax.hist(X[:,3], histtype="step", lw=0.5, ls="--", color="r")
    ax.set_xlabel("Xu Score")
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax.text(0.2, 0.9, r"$\mu_{Xu}=$"+"%.2f"%np.mean(X[:,3]), ha="left", va="center", transform=ax.transAxes)
    if gmm: fig.savefig("../outputs/histograms_ss_%s.gmm.png"%a_name, bbox_inches="tight")
    else: fig.savefig("../outputs/histograms_ss_%s.png"%a_name, bbox_inches="tight")
    utils.remove_local_files(fname)
    return

def find_peaks(ax, x, y, dist, ids, lim, rotation, dh, log, param_name="Velocity"):
    def parse_params_text(val, log=True):
        txt = "-" if val < 0 else ""
        if log: txt += "$10^{%.1f}$"%np.abs(val)
        else: txt += "%.1f"%val
        return txt
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y, distance=dist)
    lines = (x[peaks] + x[peaks+1])/2
    print(" Lines(%s):"%param_name, lines)
    for ix, l in enumerate(lines):
        if ix in ids: 
            ax.axvline(l, color="k", ls="--", lw=0.6)
            ax.text(l, lim+dh, parse_params_text(l, log), ha="center", va="center", fontdict={"size":6,"color":"g"}, rotation=rotation)
    return 

def histograms_scatters(rads, a_name, starts, ends, sctr=-1, gmm=False, case=0, kind=1, params=["v", "w_l", "p_l"],
                       fp_details_list={
                           "v": {"dist": 7, "ids":[1,2,3,4,5,6], "bins":100, "ylim":[0,0.4], "rot":45, "dh":0.03, "log":True},
                           "w_l": {"dist": 7, "ids":[1,2,3], "bins":100, "ylim":[0,1.4], "rot":45, "dh":0.07, "log":True},
                           "p": {"dist": 7, "ids":[1], "bins":100, "ylim":[0,0.4], "rot":45, "dh":0.03, "log":False}
                       }):
    det_type = {0:"Sudden [2004]", 1:"Blanchard [2006]", 2:"Blanchard [2009]"}
    types = {-1:"US", 0:"IS", 1:"GS"}
    ty = types[sctr]
    if gmm: fname = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv"
    else: fname = "../outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    X = pd.DataFrame()
    for rad, start, end in zip(rads, starts, ends):
        dn = start
        while dn <= end:
            floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
            dn = dn + dt.timedelta(days=1)
            y = pd.read_csv(floc)
            #y = y[y["gflg_%d_%d"%(case,kind)]==sctr]
            y = utils._run_riberio_threshold_on_rad(y)
            y = y[y["ribiero_gflg"]==sctr]
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
            X = pd.concat([X, y[params]])
    X = X[params].values
    fig, axes = plt.subplots(dpi=150, figsize=(9,2.5), nrows=1, ncols=3)
    ax = axes[0]
    v = X[:,0]
    v[v==0] = 1e-5
    v = np.sign(v)*np.log10(np.abs(v))
    vhist, vbins = np.histogram(v, bins=fp_details["v"]["bins"], density=True)
    #find_peaks(ax, vbins, vhist, dist=fp_details["v"]["dist"], ids=fp_details["v"]["ids"], lim=fp_details["v"]["ylim"][1],
    #          rotation=fp_details["v"]["rot"], dh=fp_details["v"]["dh"], log=fp_details["v"]["log"], param_name="Velocity")
    ax.hist(v, bins=fp_details["v"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
    ax.set_xlim(-4,4)
    ax.set_xticklabels([r"-$10^4$", r"-$10^2$", r"$10^0$", r"$10^2$", r"$10^4$"])
    ax.set_xlabel(r"Velocity, $ms^{-1}$")
    ax.set_ylabel("Density")
    ax.set_ylim(fp_details["v"]["ylim"])
    ax = axes[1]
    w = X[:,1]
    w[w==0] = 1e-5
    w = np.sign(w)*np.log10(np.abs(w))
    whist, wbins = np.histogram(w, bins=fp_details["w_l"]["bins"], density=True)
    #find_peaks(ax, wbins, whist, dist=fp_details["w_l"]["dist"], ids=fp_details["w_l"]["ids"], lim=fp_details["w_l"]["ylim"][1],
    #          rotation=fp_details["w_l"]["rot"], dh=fp_details["w_l"]["dh"], log=fp_details["w_l"]["log"], param_name="Spect. Width")
    ax.hist(w, bins=fp_details["w_l"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
    ax.set_xlabel(r"Spect. Width, $ms^{-1}$")
    ax.set_xlim(-2,4)
    ax.set_ylim(fp_details["w_l"]["ylim"])
    ax.set_xticklabels([r"-$10^2$", r"-$10^0$", r"$10^2$", r"$10^4$"])
    ax = axes[2]
    p = X[:,2]
    ax.text(1.02, 0.1, "Type: "+ty, ha="left", va="center", transform=ax.transAxes, rotation=90, fontdict={"size":7,"color":"r"})
    ax.text(1.02, 0.8, "ACFs~ %dK"%(len(v)/1000), ha="left", va="center", transform=ax.transAxes, rotation=90, fontdict={"size":7,"color":"r"})
    ax.text(0.5, 1.05, "GS: "+det_type[case], ha="left", va="center", transform=ax.transAxes, fontdict={"size":7,"color":"r"})
    phist, pbins = np.histogram(p, bins=fp_details["p"]["bins"], density=True)
    #find_peaks(ax, pbins, phist, dist=fp_details["p"]["dist"], ids=fp_details["p"]["ids"], lim=fp_details["p"]["ylim"][1],
    #          rotation=fp_details["p"]["rot"], dh=fp_details["p"]["dh"], log=fp_details["p"]["log"], param_name="Power")
    ax.hist(p, bins=fp_details["p"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
    ax.set_xlabel("Power, dB")
    ax.set_xlim([0,20])
    ax.set_ylim(fp_details["p"]["ylim"])
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    if gmm: fig.savefig("../outputs/histograms_scat_%s.gmm_%s_%d_%d.png"%(a_name,ty,case,kind), bbox_inches="tight")
    else: fig.savefig("../outputs/histograms_scat_%s_%s_%d_%d.png"%(a_name,ty,case,kind), bbox_inches="tight")
    return



def histograms_scatters_from_remote(rads, dates, a_name, sctrs=[-1, 0, 1], gmm=False, case=0, kind=0, params=["v", "w_l", "p_l"],
                       fp_details_list=[], png_fname="", gates=10):
    det_type = {0:"Sudden [2004]", 1:"Blanchard [2006]", 2:"Blanchard [2009]"}
    types = {-1:"US", 0:"IS", 1:"GS"}
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    if gmm: fname = "../outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    else: fname = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv"
    fig, axes = plt.subplots(dpi=150, figsize=(10, 10), nrows=3, ncols=3)
    for rad, dn in zip(rads, dates):
        floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
        utils.fetch_file(conn, floc, LFS)
    
    for ix, sctr in enumerate(sctrs):
        if ix >= 0:
            fp_details = fp_details_list[ix]
            X = pd.DataFrame()
            for rad, dn in zip(rads, dates):
                floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
                if os.path.exists(floc):
                    y = pd.read_csv(floc)
                    y = y[(y["gflg_%d_%d"%(case,kind)]==sctr) & (y.slist>=gates)]
                    y = y.replace([np.inf, -np.inf], np.nan).dropna()
                    X = pd.concat([X, y[params]])
            X = X[params].values
            ty = types[sctr]
            ax = axes[ix, 0]
            v = X[:,0]
            v[v==0] = 1e-5
            v = np.sign(v)*np.log10(np.abs(v))
            vhist, vbins = np.histogram(v, bins=fp_details["v"]["bins"], density=True)
            find_peaks(ax, vbins, vhist, dist=fp_details["v"]["dist"], ids=fp_details["v"]["ids"], lim=fp_details["v"]["ylim"][1],
                      rotation=fp_details["v"]["rot"], dh=fp_details["v"]["dh"], log=fp_details["v"]["log"], param_name="Velocity")
            ax.hist(v, bins=fp_details["v"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlim(-4,4)
            ax.set_xticks([-4,-2,0,2,4])
            ax.set_xticklabels([r"-$10^4$", r"-$10^2$", r"$10^0$", r"$10^2$", r"$10^4$"])
            ax.set_xlabel(r"Velocity, $ms^{-1}$")
            ax.set_ylabel("Density")
            ax.set_ylim(fp_details["v"]["ylim"])

            ax = axes[ix, 1]
            w = X[:,1]
            w[w==0] = 1e-5
            w = np.sign(w)*np.log10(np.abs(w))
            whist, wbins = np.histogram(w, bins=fp_details["w_l"]["bins"], density=True)
            find_peaks(ax, wbins, whist, dist=fp_details["w_l"]["dist"], ids=fp_details["w_l"]["ids"], lim=fp_details["w_l"]["ylim"][1],
                      rotation=fp_details["w_l"]["rot"], dh=fp_details["w_l"]["dh"], log=fp_details["w_l"]["log"], 
                       param_name="Spect. Width")
            ax.hist(w, bins=fp_details["w_l"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlabel(r"Spect. Width, $ms^{-1}$")
            ax.set_xlim(-2,4)
            ax.set_ylim(fp_details["w_l"]["ylim"])
            ax.set_xticks([-2,0,2,4])
            ax.set_xticklabels([r"-$10^2$", r"-$10^0$", r"$10^2$", r"$10^4$"])

            ax = axes[ix, 2]
            p = X[:,2]
            ax.text(1.02, 0.1, "Type: "+ty, ha="left", va="center", transform=ax.transAxes, rotation=90, fontdict={"size":7,"color":"r"})
            ax.text(1.02, 0.8, "ACFs~ %dK"%(len(v)/1000), ha="left", va="center", transform=ax.transAxes, rotation=90, 
                    fontdict={"size":7,"color":"r"})
            if ix==0: ax.text(0.5, 1.05, "GS: "+det_type[case], ha="left", va="center", transform=ax.transAxes, 
                              fontdict={"size":7,"color":"r"})
            phist, pbins = np.histogram(p, bins=fp_details["p"]["bins"], density=True)
            find_peaks(ax, pbins, phist, dist=fp_details["p"]["dist"], ids=fp_details["p"]["ids"], lim=fp_details["p"]["ylim"][1],
                  rotation=fp_details["p"]["rot"], dh=fp_details["p"]["dh"], log=fp_details["p"]["log"], param_name="Power")
            ax.hist(p, bins=fp_details["p"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlabel("Power, dB")
            ax.set_xlim([0,20])
            ax.set_ylim(fp_details["p"]["ylim"])
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.savefig("../outputs/" + png_fname, bbox_inches="tight")
    os.system("rm -rf ../outputs/cluster_tags/*")
    return

def histograms_indp_scatters_from_remote(rads, dates, a_name, sctrs=[0, 1], gmm=False, case=0, kind=0, params=["v", "w_l", "p_l"],
                       fp_details_list=[], png_fname=""):
    det_type = {0:"Sudden [2004]", 1:"Blanchard [2006]", 2:"Blanchard [2009]"}
    types = {-1:"US", 0:"IS", 1:"GS"}
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    if gmm: fname = "../outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    else: fname = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv"
    fig, axes = plt.subplots(dpi=150, figsize=(10, 7), nrows=2, ncols=3)
    for rad, dn in zip(rads, dates):
        floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
        utils.fetch_file(conn, floc, LFS)
    
    for ix, sctr in enumerate(sctrs):
        if ix >= 0:
            fp_details = fp_details_list[ix]
            X = pd.DataFrame()
            for rad, dn in zip(rads, dates):
                floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
                if os.path.exists(floc):
                    y = pd.read_csv(floc)
                    y["classical"] = utils.ScatterTypeDetection.indp_classical(y.w_l, y.v)
                    y = y[y.classical==sctr]
                    y = y.replace([np.inf, -np.inf], np.nan).dropna()
                    X = pd.concat([X, y[params]])
            X = X[params].values
            ty = types[sctr]
            ax = axes[ix, 0]
            v = X[:,0]
            v[v==0] = 1e-5
            v = np.sign(v)*np.log10(np.abs(v))
            vhist, vbins = np.histogram(v, bins=fp_details["v"]["bins"], density=True)
            find_peaks(ax, vbins, vhist, dist=fp_details["v"]["dist"], ids=fp_details["v"]["ids"], lim=fp_details["v"]["ylim"][1],
                      rotation=fp_details["v"]["rot"], dh=fp_details["v"]["dh"], log=fp_details["v"]["log"], param_name="Velocity")
            ax.hist(v, bins=fp_details["v"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlim(-4,4)
            ax.set_xticks([-4,-2,0,2,4])
            ax.set_xticklabels([r"-$10^4$", r"-$10^2$", r"$10^0$", r"$10^2$", r"$10^4$"])
            ax.set_xlabel(r"Velocity, $ms^{-1}$")
            ax.set_ylabel("Density")
            ax.set_ylim(fp_details["v"]["ylim"])

            ax = axes[ix, 1]
            w = X[:,1]
            w[w==0] = 1e-5
            w = np.sign(w)*np.log10(np.abs(w))
            whist, wbins = np.histogram(w, bins=fp_details["w_l"]["bins"], density=True)
            find_peaks(ax, wbins, whist, dist=fp_details["w_l"]["dist"], ids=fp_details["w_l"]["ids"], lim=fp_details["w_l"]["ylim"][1],
                      rotation=fp_details["w_l"]["rot"], dh=fp_details["w_l"]["dh"], log=fp_details["w_l"]["log"], 
                       param_name="Spect. Width")
            ax.hist(w, bins=fp_details["w_l"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlabel(r"Spect. Width, $ms^{-1}$")
            ax.set_xlim(-2,4)
            ax.set_ylim(fp_details["w_l"]["ylim"])
            ax.set_xticks([-2,0,2,4])
            ax.set_xticklabels([r"-$10^2$", r"-$10^0$", r"$10^2$", r"$10^4$"])

            ax = axes[ix, 2]
            p = X[:,2]
            ax.text(1.02, 0.1, "Type: "+ty, ha="left", va="center", transform=ax.transAxes, rotation=90, fontdict={"size":7,"color":"r"})
            ax.text(1.02, 0.8, "ACFs~ %dK"%(len(v)/1000), ha="left", va="center", transform=ax.transAxes, rotation=90, 
                    fontdict={"size":7,"color":"r"})
            if ix==0: ax.text(0.5, 1.05, "GS: "+det_type[case], ha="left", va="center", transform=ax.transAxes, 
                              fontdict={"size":7,"color":"r"})
            phist, pbins = np.histogram(p, bins=fp_details["p"]["bins"], density=True)
            find_peaks(ax, pbins, phist, dist=fp_details["p"]["dist"], ids=fp_details["p"]["ids"], lim=fp_details["p"]["ylim"][1],
                  rotation=fp_details["p"]["rot"], dh=fp_details["p"]["dh"], log=fp_details["p"]["log"], param_name="Power")
            ax.hist(p, bins=fp_details["p"]["bins"], histtype="step", lw=0.9, ls="-", color="b", density=True)
            ax.set_xlabel("Power, dB")
            ax.set_xlim([0,20])
            ax.set_ylim(fp_details["p"]["ylim"])
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.savefig("../outputs/" + png_fname, bbox_inches="tight")
    os.system("rm -rf ../outputs/cluster_tags/*")
    return

def plot_2D_hist(rads, dates, a_name, gmm, case, kind, png_fname):
    def to_2dhist(df, zparam, val, xparam="umin", yparam="doy"):
        df = df[ [xparam, yparam, zparam] ]
        df = df[df[zparam]==val]
        df = df.groupby( [xparam, yparam] ).count().reset_index()
        df = df[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
        _x = df.index.values
        _y = df.columns.levels[1].values
        _xx, _yy  = np.meshgrid( _x, _y )
        _zz = np.ma.masked_where(np.isnan(df[zparam].values), df[zparam].values)
        return _xx, _yy, _zz, _x, _y
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    if gmm: fname = "../outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    else: fname = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv"
    fig, axes = plt.subplots(dpi=150, figsize=(10, 10), nrows=2, ncols=2)
    X = pd.DataFrame()
    dumin, ix = 3, 0
    suntime = {"doy":[], "sunrise":[], "sunset":[]}
    for rad, dn in zip(rads, dates):
        floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
        if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
        if os.path.exists(floc):
            hdw = pydarn.read_hdw_file(rad)
            o = Observer(hdw.geographic.lat, hdw.geographic.lon)
            s = sun(o, date=dn)
            suntime["doy"].append(dn.dayofyear)
            suntime["sunrise"].append(s["sunrise"].hour + (dumin/60.)*int(s["sunrise"].minute/dumin))
            suntime["sunset"].append(s["sunset"].hour + (dumin/60.)*int(s["sunset"].minute/dumin))
            u = pd.read_csv(floc)
            u = utils._run_riberio_threshold_on_rad(u)
            u.time = u.time.apply(lambda t: num2date(t, tz=pytz.timezone("UTC")))
            u["ut"], u["doy"], u["umin"] = u.time.dt.hour, u.time.dt.dayofyear,\
                        u.time.apply(lambda t: t.hour + (dumin/60.)*int(t.minute/dumin))
            X = pd.concat([X, u])
            #os.system("rm -rf ../outputs/cluster_tags/*")
        #if ix == 3: break
        ix += 1
    os.system("rm -rf ../outputs/cluster_tags/*")
    X = X[["time", "slist", "bmnum", "trad_gsflg", "gflg_%d_%d"%(case,kind), "ut", "doy", "umin", "ribiero_gflg"]]
    ax = axes[0,0]
    ax.set_xlabel("UT")
    ax.set_ylabel("DoY")
    ax.text(0.1,1.03,"IS: fitacf",ha="left",va="center",transform=ax.transAxes)
    _xx, _yy, _zz, _x, _y = to_2dhist(X, "trad_gsflg", 0)
    ax.pcolormesh(_xx, _yy, _zz.T, cmap="jet")
    ax.plot(suntime["sunrise"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax.plot(suntime["sunset"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax = axes[1,0]
    ax.set_xlabel("UT")
    ax.set_ylabel("DoY")
    ax.text(0.1,1.03,"GS: fitacf",ha="left",va="center",transform=ax.transAxes)
    _xx, _yy, _zz, _x, _y = to_2dhist(X, "trad_gsflg", 1)
    ax.pcolormesh(_xx, _yy, _zz.T, cmap="jet")
    ax.plot(suntime["sunrise"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax.plot(suntime["sunset"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax = axes[0,1]
    ax.set_xlabel("UT")
    ax.set_ylabel("DoY")
    ax.text(0.1,1.03,"IS: Riberio.11",ha="left",va="center",transform=ax.transAxes)
    _xx, _yy, _zz, _x, _y = to_2dhist(X, "ribiero_gflg", 0)#"gflg_%d_%d"%(case,kind)
    ax.pcolormesh(_xx, _yy, _zz.T, cmap="jet")
    ax.plot(suntime["sunrise"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax.plot(suntime["sunset"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax = axes[1,1]
    ax.set_xlabel("UT")
    ax.set_ylabel("DoY")
    ax.text(0.1,1.03,"GS: Riberio.11",ha="left",va="center",transform=ax.transAxes)
    _xx, _yy, _zz, _x, _y = to_2dhist(X, "ribiero_gflg", 1)
    ax.pcolormesh(_xx, _yy, _zz.T, cmap="jet")
    ax.plot(suntime["sunrise"], suntime["doy"], color="k", lw=1.5, ls="--")
    ax.plot(suntime["sunset"], suntime["doy"], color="k", lw=1.5, ls="--")
    fig.suptitle(png_fname.replace(".png", "").replace("_", " "))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.savefig("../outputs/" + png_fname, bbox_inches="tight")
    return