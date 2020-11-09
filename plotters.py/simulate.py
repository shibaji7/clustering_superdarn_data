
import numpy as np
import sys
sys.path.insert(0, '..')
from algorithms.dbscan_gmm import DBSCAN_GMM
from algorithms.grid_based_dbscan import GridBasedDBSCAN
from algorithms.grid_based_dbscan_gmm import GridBasedDBSCAN_GMM
from utilities.plot_utils import *
from utility import Skills, ScatterDetection
import pandas as pd
import datetime
from plots_report import *

def estimate_skills(_dict_, labels):
    V, W, L = [], [], []
    for v, w, l in zip(_dict_["vel"], _dict_["wid"], labels):
        V.extend(v.tolist())
        W.extend(w.tolist())
        L.extend(l.tolist())
    V, W, L = np.array(V), np.array(W), np.array(L)
    X = np.array([V.tolist(), W.tolist()]).T
    sk = Skills(X, L)
    return sk


case = 3

if case == 0:
    start_time = datetime.datetime(2017, 4, 4)
    end_time = datetime.datetime(2017, 4, 5)
    rad, bm = "cvw",7

    start_time = datetime.datetime(2015, 3, 17)
    end_time = datetime.datetime(2015, 3, 17, 12)
    rad, bm = "bks",15

    db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
    setattr(db, "skill", estimate_skills(db.data_dict, db.clust_flg))
    dbgmm = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True)
    setattr(dbgmm, "skill", estimate_skills(dbgmm.data_dict, dbgmm.clust_flg))
    gbdb = GridBasedDBSCAN(start_time, end_time, rad, load_model=False, save_model=True)
    setattr(gbdb, "skill", estimate_skills(gbdb.data_dict, gbdb.clust_flg))
    gbdbgmm = GridBasedDBSCAN_GMM(start_time, end_time, rad, load_model=False, save_model=True)
    setattr(gbdbgmm, "skill", estimate_skills(gbdbgmm.data_dict, gbdbgmm.clust_flg))

    rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=4)
    rti.addClusterPlot(db.data_dict, db.clust_flg, bm, "DBSCAN", label_clusters=True, skill=db.skill)
    rti.addClusterPlot(dbgmm.data_dict, dbgmm.clust_flg, bm, "DBSCAN + GMM", label_clusters=True, skill=dbgmm.skill)
    rti.addClusterPlot(gbdb.data_dict, gbdb.clust_flg, bm, "GB-DBSCAN", label_clusters=True, skill=gbdb.skill)
    rti.addClusterPlot(gbdbgmm.data_dict, gbdbgmm.clust_flg, bm, "GB-DBSCAN + GMM ", label_clusters=True, xlabel="Time, UT", 
        skill=gbdbgmm.skill)
    rti.save("figs/rti.example.png")
if case == 1:
    plot_acfs(rad="kap")
    plot_lims(False)
    plot_lims(True)
    plot_rad_acfs()
    plot_hist_hr()
if case == 2:
    start_time = datetime.datetime(2015, 3, 17)
    end_time = datetime.datetime(2015, 3, 17, 12)
    rad, bm = "bks",15
    #start_time = datetime.datetime(2017, 4, 4)
    #end_time = datetime.datetime(2017, 4, 5)
    #rad, bm = "cvw",7

    db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
    rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=3)
    rti.addClusterPlot(db.data_dict, db.clust_flg, bm, "DBSCAN", label_clusters=True, skill=None)
    rti.addGSISPlot(db.data_dict, db.data_dict["trad_gsflg"], bm, "GS-ID:Traditioanl", show_closerange=True, xlabel='')
    rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
    rti.save("figs/dbscan.trad.png")
if case == 3:
    start_time = datetime.datetime(2015, 3, 17)
    end_time = datetime.datetime(2015, 3, 17, 12)
    rad, bm = "bks",15
    #start_time = datetime.datetime(2017, 4, 4)
    #end_time = datetime.datetime(2017, 4, 5)
    #rad, bm = "cvw",7
    kinds = ["dbscan", "dbscan-gmm", "gb-dbscan", "gb-dbscan-gmm"]
    kinds = ["gb-dbscan-gmm"]
    for kind in kinds:
    
        if kind == "dbscan": db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
        if kind == "dbscan-gmm": db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=True)
        if kind == "gb-dbscan": db = GridBasedDBSCAN(start_time, end_time, rad, load_model=False, save_model=True)
        if kind == "gb-dbscan-gmm": db = GridBasedDBSCAN_GMM(start_time, end_time, rad, load_model=False, save_model=True,
                features=['beam', 'gate', 'time'], scan_eps=10)
        sd = ScatterDetection(db.data_dict)
        rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=0), bm, "GS-ID:Median(Sudden)", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=1), bm, "GS-ID:Median(Blanchard 2006)", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=2), bm, "GS-ID:Median(Blanchard 2009)", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=3), bm, "GS-ID:Median(Proposed)", show_closerange=True, xlabel='')
        rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        rti.save("figs/%s.median.png"%kind)

        rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=0), bm, "GS-ID:Median(Sudden)", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=1), bm, "GS-ID:Median(Blanchard 2006)",
                show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=2), bm, "GS-ID:Median(Blanchard 2009)", 
                show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=2, case=3), bm, "GS-ID:Median(Proposed)", show_closerange=True, xlabel='')
        rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        rti.save("figs/%s.kde.png"%kind)

        rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=0), bm, "GS-ID:Sudden", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=1), bm, "GS-ID:Blanchard 2006", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=2), bm, "GS-ID:Blanchard 2009", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=3), bm, "GS-ID:Proposed", show_closerange=True, xlabel='')
        rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        rti.save("figs/%s.indp.png"%kind)
