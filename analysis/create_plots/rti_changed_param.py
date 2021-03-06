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

import pydarn
from pysolar.solar import get_altitude

from plotlib import RangeTimePlot as RTI
import utils
import rad_fov

LFS = "LFS/LFS_clustering_superdarn_data/"

def get_sza(row, lats, lons):
    print(num2date(row.time), row.bmnum)
    lat, lon = lats[int(row["bmnum"]), int(row["slist"])],\
                    lons[int(row["bmnum"]), int(row["slist"])]
    dn = num2date(row["time"])
    d = dn.replace(tzinfo=dt.timezone.utc)
    sza = 90.-get_altitude(lat, lon, d)
    return sza

def _add_axis(fig, val):
    ax = fig.add_subplot(val[0],val[1],val[2])
    ax.tick_params(axis="both", labelsize=12)
    return ax

gmm = True
a_name = "dbscan"
rads = ["cvw", "cvw"]
dates = [dt.datetime(2012,1,5), dt.datetime(2012,1,5)]
beam = 14
remove_file = False
maxGate = None
tags = [True, False]

pubfile = utils.get_pubfile()
conn = utils.get_session(key_filename=pubfile)

X = {}
i = 0
for rad, dn, tag in zip(rads, dates, tags):
    if gmm: fname = "../outputs/figures_for_papers/{rad}.{a_name}.gmm.{dn}.csv"
    else: fname = "../outputs/figures_for_papers/{rad}.{a_name}.{dn}.csv"
    if tag: fname = fname.replace(".csv", ".missing.csv")
    floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
    print(floc)
    if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
    key = rad+"%d"%i
    X[key] = pd.read_csv(floc)
    hdw = pydarn.read_hdw_file(rad)
    egate = hdw.gates if not maxGate else maxGate
    rfov = rad_fov.CalcFov(hdw=hdw, ngates=egate)
    if "sza" not in X[key].columns: 
        X[key]["sza"] = X[key].apply(lambda r: get_sza(r, rfov.latFull, rfov.lonFull), axis=1)
        X[key].to_csv(floc)
    X[key] = utils._run_riberio_threshold_on_rad(X[key])
    i += 1

conn.close()
if remove_file: os.system("rm -rf ../outputs/cluster_tags/*")

fig = plt.figure(figsize=(16, 15), dpi=100)
mpl.rcParams.update({"font.size": 10})

x = X["cvw0"]
rti = RTI(75, np.unique(x.time), "", num_subplots=3)
rti.addParamPlot(x, beam, "Date: %s, %s, Beam: %d"%(dates[0].strftime("%Y-%m-%d"), rads[0].upper(), beam) + "\n" +\
                 r"$\lbrace\epsilon_{B}, \epsilon_{RG}, \epsilon_{S}, C_{GMM}\rbrace=\lbrace3,1,1,5\rbrace$", 
                 p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]",
                 ax=_add_axis(fig, (5,2,1)), fig=fig, addcb=False)
rti.addParamPlot(x, beam, "", p_max=33, p_min=3, p_step=6, xlabel="", zparam="p_l", label="Power [db]",
                 ax=_add_axis(fig, (5,2,3)), fig=fig, addcb=False)
rti.addParamPlot(x, beam, "", p_max=50, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spect. Width [m/s]", 
                 ax=_add_axis(fig, (5,2,5)), fig=fig, addcb=False)
rti.addCluster(x, beam, "", label_clusters=True, skill=None, xlabel="", ax=_add_axis(fig, (5,2,7)))
rti.addGSIS(x, beam, "", xlabel="Time, UT", zparam="ribiero_gflg", ax=_add_axis(fig, (5,2,9)))

x = X["cvw1"]
x = utils._run_riberio_threshold(x, beam)
rti = RTI(75, np.unique(x.time), "", num_subplots=3)
rti.addParamPlot(x, beam, "\n" + r"$\lbrace\epsilon_{B}, \epsilon_{RG}, \epsilon_{S}, C_{GMM}\rbrace=\lbrace1,1,1,10\rbrace$", 
                 p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]", ax=_add_axis(fig, (5,2,2)), fig=fig)
rti.addParamPlot(x, beam, "", p_max=33, p_min=3, p_step=6, xlabel="", zparam="p_l", label="Power [db]",
                 ax=_add_axis(fig, (5,2,4)), fig=fig)
rti.addParamPlot(x, beam, "", p_max=50, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spect. Width [m/s]", 
                 ax=_add_axis(fig, (5,2,6)), fig=fig)
rti.addCluster(x, beam, "", label_clusters=True, skill=None, xlabel="", ax=_add_axis(fig, (5,2,8)))
rti.addGSIS(x, beam, "", xlabel="Time, UT", zparam="ribiero_gflg", ax=_add_axis(fig, (5,2,10)))

fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.autofmt_xdate()
fig.savefig("create_plots/images/paramstudy.png", bbox_inches="tight")