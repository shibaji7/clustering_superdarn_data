import os
import sys
sys.path.append("./")
sys.path.append("create_plots/")
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.dates import num2date
import matplotlib.pyplot as plt
from scipy.stats import beta

import pydarn
from pysolar.solar import get_altitude
import utils
import rad_fov

################################################
# Inputs date and radar name
################################################
Rad, Dn = "cvw", dt.datetime(2012,1,2)

LFS = "LFS/LFS_clustering_superdarn_data/"
gmm = True
a_name = "dbscan"
rads = [Rad]
dates = [Dn]
remove_file = False
maxGate = None
gate = 7
sza_thresh, wid = {"is":85., "gs":105.}, 1.
bin_thick = 1.
tag = False
scale = 1.

def get_sza(row, lats, lons):
    print(num2date(row.time), row.bmnum)
    lat, lon = lats[int(row["bmnum"]), int(row["slist"])],\
                    lons[int(row["bmnum"]), int(row["slist"])]
    dn = num2date(row["time"])
    d = dn.replace(tzinfo=dt.timezone.utc)
    sza = 90.-get_altitude(lat, lon, d)
    return sza

def calculate_total_uncertainity(_is, _gs):
    
    def calc_prob(m, l, th, low=True):
        h, be = np.histogram(m, bins=l, density=True)
        bc = np.diff(be)
        idx = be[1:] < th if low else be[1:] > th
        edges = (be[1:])[be[1:] < th] if low else (be[1:])[be[1:] > th]
        pr = np.sum(h[idx]*bc[idx])
        height = h[idx]
        #return pr, h, be[1:], bc
        return pr, height, edges, bc
    
    L = len(np.unique(_is.sza)) if len(np.unique(_is.sza)) > len(np.unique(_gs.sza)) else len(np.unique(_gs.sza))
    L = int(L/bin_thick)
    pr_a, h_a, be_a, bc = calc_prob(_is.sza.tolist(), L, sza_thresh["is"], low=True)
    pr_b, h_b, be_b, _ = calc_prob(_gs.sza.tolist(), L, sza_thresh["gs"], low=False)
    out = []
    for pa, pb, b in zip(h_a, h_b, bc):
        if pa < pb: out.append(pa*b)
        else: out.append(pb*b)
    #prb = np.round(np.sum(out), 3)
    prb = np.round(pr_a+pr_b, 3)
    print("Total Uncertainty:", prb)
    fig = plt.figure(figsize=(4, 4), dpi=100)
    mpl.rcParams.update({"font.size": 10})
    ax = fig.add_subplot(111)
    ax.hist(_is.sza.tolist(), bins=L, histtype="step", color="red", density=True, alpha=0.5, label="IS")
    ax.hist(_gs.sza.tolist(), bins=L, histtype="step", color="blue", density=True, alpha=0.5, label="GS")
    ax.fill_between(be_a, y1=np.zeros(len(be_a)), y2=h_a, color="r", alpha=0.3, step="pre")
    ax.fill_between(be_b, y1=np.zeros(len(be_b)), y2=h_b, color="b", alpha=0.3, step="pre")
    #ax.fill_between(be_a, y1=np.zeros(len(be_a)), y2=out, color="violet", alpha=0.3, step="pre")
    ax.legend(loc=1)
    ax.axvline(sza_thresh["is"], color="r", ls="--", lw=0.6)
    ax.axvline(sza_thresh["gs"], color="b", ls="--", lw=0.6)
    ax.set_xlabel(r"SZA, $\chi$ ($^o$)")
    ax.set_ylabel("Density of IS, GS")
    ax.text(0.99, 1.03, "Prob.~ %.2f"%prb, ha="right", va="center", transform=ax.transAxes)
    ax.text(0.01, 1.03, rads[0].upper()+", %s"%dates[0].strftime("%Y-%m-%d"), ha="left", va="center", transform=ax.transAxes)
    png = "create_plots/images/detection_%s_%s.png"%(Rad, Dn.strftime("%Y%m%d"))
    ax.set_ylim(0,.1)
    if tag: png = png.replace(".png", ".missing.png")
    fig.savefig(png, bbox_inches="tight")
    return prb

pubfile = utils.get_pubfile()
conn = utils.get_session(key_filename=pubfile)

if gmm: fname = "../outputs/figures_for_papers/{rad}.{a_name}.gmm.{dn}.csv"
else: fname = "../outputs/figures_for_papers/{rad}.{a_name}.{dn}.csv"
    
if tag: fname = fname.replace(".csv", ".missing.csv")
    
for rad, dn in zip(rads, dates):
    floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
    if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
    X = pd.read_csv(floc)
    hdw = pydarn.read_hdw_file(rad)
    egate = hdw.gates if not maxGate else maxGate
    rfov = rad_fov.CalcFov(hdw=hdw, ngates=egate)
    if "sza" not in X.columns: 
        X["sza"] = X.apply(lambda r: get_sza(r, rfov.latFull, rfov.lonFull), axis=1)
        X.to_csv(floc)
    X = utils._run_riberio_threshold_on_rad(X)
conn.close()
if remove_file: os.system("rm -rf ../outputs/cluster_tags/*")

X.sza = (np.array(X.sza)/wid).astype(int)*wid
X = X[X.slist > gate]
X = X[["sza", "ribiero_gflg"]]
_is, _gs = X[X.ribiero_gflg==0], X[X.ribiero_gflg==1]
Pr_ab = calculate_total_uncertainity(_is, _gs)