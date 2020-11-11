#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from get_sd_data import *
from matplotlib.dates import date2num, num2date

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

def estimate_df_skills(df, labels):
    V, W, L = [], [], []
    V, W, L = np.array(df.v), np.array(df.w_l), np.array(df.labels)
    X = np.array([V.tolist(), W.tolist()]).T
    sk = Skills(X, L)
    return sk

def _filter_by_time(start_time, end_time, data_dict):
    time = data_dict['time']
    start_i, end_i = None, None
    start_time, end_time = date2num(start_time), date2num(end_time)
    if start_time < time[0][0]: # Sometimes start time is a few seconds before the first scan
        start_time = time[0][0]
    for i, t in enumerate(time):
        if np.sum(start_time >= t) > 0 and start_i == None:
            start_i = i
        if np.sum(end_time > t) > 0 and start_i != None:
            end_i = i+1
    data_dict['gate'] = data_dict['gate'][start_i:end_i]
    data_dict['time'] = data_dict['time'][start_i:end_i]
    data_dict['beam'] = data_dict['beam'][start_i:end_i]
    data_dict['vel'] = data_dict['vel'][start_i:end_i]
    data_dict['wid'] = data_dict['wid'][start_i:end_i]
    data_dict['elv'] = data_dict['elv'][start_i:end_i]
    data_dict['trad_gsflg'] = data_dict['trad_gsflg'][start_i:end_i]
    return data_dict

def todf(dicts, keys=['gate', 'beam', 'vel', 'wid', 'time', 'trad_gsflg', 'elv', 'pow', 'clust_flg']):
    df = pd.DataFrame()
    _o = {}
    print(dicts.keys())
    for k in keys:
        _o[k] = []
        for x in dicts[k]:
            _o[k].extend(x)
    df = pd.DataFrame.from_records(_o)
    df = df.rename(columns={'gate':"slist", 'beam':"bmnum", 'vel':'v', 'wid':"w_l", 
                            'time':"time", 'pow':"p_l", 'clust_flg':"labels"})
    return df

def sma_bbox(scans, sdur=5, idx=None, dbeam=15, window=7):
    df = pd.DataFrame()
    plot=False
    for i in range(int(len(scans)/sdur)):
        if (idx is not None) and (i == idx): plot=True
        if i == 0: mlf = MiddleLatFilter(rad, scans=scans[i*sdur:(i+1)*sdur], plot=plot)
        elif i == int(len(scans)/sdur)-1: mlf._reset_(rad, scans[i*sdur:], plot=plot)
        else: mlf._reset_(rad, scans[i*sdur:(i+1)*sdur], plot=plot)
        dx = mlf.doFilter(fdata, dbeam=dbeam, window=window)
        slist = np.array(dx.slist)
        labs = np.array(dx["labels"])
        labs[labs<0] = np.nan
        labs = labs + (10*i)
        labs[np.isnan(labs)] = -1
        dx["labels"] = labs
        df = pd.concat([df, dx])
    return df

def lower_range(df, gf=None):
    u = df.copy()
    slist = np.array(u.slist)
    labs = np.array(u["labels"])
    if gf is not None: labs[slist<8] = gf
    u["labels"] = labs
    return u


# In[2]:


case = 0

if case == 0:
    start_time = datetime.datetime(2017, 4, 4)
    end_time = datetime.datetime(2017, 4, 5)
    rad, bm = "cvw",7

    #db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
    #setattr(db, "skill", estimate_skills(db.data_dict, db.clust_flg))
    dbgmm = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True)
    setattr(dbgmm, "skill", estimate_skills(dbgmm.data_dict, dbgmm.clust_flg))
    #gbdb = GridBasedDBSCAN(start_time, end_time, rad, load_model=False, save_model=True)
    #setattr(gbdb, "skill", estimate_skills(gbdb.data_dict, gbdb.clust_flg))
    #gbdbgmm = GridBasedDBSCAN_GMM(start_time, end_time, rad, load_model=False, save_model=True)
    #setattr(gbdbgmm, "skill", estimate_skills(gbdbgmm.data_dict, gbdbgmm.clust_flg))

    rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=7)
    dx = todf(db.data_dict)
    #rti.addParamPlot(dx, bm, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label='Velocity [m/s]')
    #rti.addParamPlot(dx, bm, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label='Power [dB]')
    #rti.addParamPlot(dx, bm, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label='Spec. Width [m/s]')
    rti.addCluster(dx, bm, "DBSCAN", label_clusters=True, skill=db.skill)
    rti.addCluster(todf(dbgmm.data_dict), bm, "DBSCAN + GMM", label_clusters=True, skill=dbgmm.skill)
    #rti.addCluster(todf(gbdb.data_dict), bm, "GB-DBSCAN", label_clusters=True, skill=gbdb.skill)
    #rti.addCluster(todf(gbdbgmm.data_dict), bm, "GB-DBSCAN + GMM ", label_clusters=True, xlabel="Time, UT", 
    #    skill=gbdbgmm.skill)
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

    db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
    rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=3)
    rti.addClusterPlot(db.data_dict, db.clust_flg, bm, "DBSCAN", label_clusters=True, skill=None)
    rti.addGSISPlot(db.data_dict, db.data_dict["trad_gsflg"], bm, "GS-ID:Traditioanl", show_closerange=True, xlabel='')
    rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
    rti.save("figs/dbscan.trad.png")
if case == 3:
    start_time = datetime.datetime(2015, 3, 17)
    end_time = datetime.datetime(2015, 3, 17, 12)
    rad, bm = "bks",7
    kinds = ["dbscan", "dbscan-gmm", "gb-dbscan", "gb-dbscan-gmm"]
    kinds = ["dbscan"]
    for kind in kinds:    
        if kind == "dbscan": db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=False)
        if kind == "dbscan-gmm": db = DBSCAN_GMM(start_time, end_time, rad, BoxCox=True, load_model=False, save_model=True, run_gmm=True)
        if kind == "gb-dbscan": db = GridBasedDBSCAN(start_time, end_time, rad, load_model=False, save_model=True)
        if kind == "gb-dbscan-gmm": db = GridBasedDBSCAN_GMM(start_time, end_time, rad, load_model=False, save_model=True,
                features=['beam', 'gate', 'time','vel','wid'], scan_eps=10)
        sd = ScatterDetection(db.data_dict)
        #rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        #rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        #rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=0), bm, "GS-ID:Median(Sudden)", show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=1), bm, "GS-ID:Median(Blanchard 2006)", show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=2), bm, "GS-ID:Median(Blanchard 2009)", show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=1, case=3), bm, "GS-ID:Median(Proposed)", show_closerange=True, xlabel='')
        #rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        #rti.save("figs/%s.median.png"%kind)

        #rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        #rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        #rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=0), bm, "GS-ID:Median(Sudden)", show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=1), bm, "GS-ID:Median(Blanchard 2006)",
        #        show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=2, thresh=[0.1,0.9], case=2), bm, "GS-ID:Median(Blanchard 2009)", 
        #        show_closerange=True, xlabel='')
        #rti.addGSISPlot(db.data_dict, sd.run(kind=2, case=3), bm, "GS-ID:Median(Proposed)", show_closerange=True, xlabel='')
        #rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        #rti.save("figs/%s.kde.png"%kind)

        rti = RangeTimePlot(110, np.unique(np.hstack(db.data_dict["time"])), "", num_subplots=6)
        rti.addClusterPlot(db.data_dict, db.clust_flg, bm, kind.upper(), label_clusters=True, skill=None)
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=0), bm, "GS-ID:Sudden", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=1), bm, "GS-ID:Blanchard 2006", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=2), bm, "GS-ID:Blanchard 2009", show_closerange=True, xlabel='')
        rti.addGSISPlot(db.data_dict, sd.run(kind=0, case=3), bm, "GS-ID:Proposed", show_closerange=True, xlabel='')
        rti.addVelPlot(db.data_dict, bm, "Velocity", vel_max=200, vel_step=50, xlabel='Time UT')
        rti.save("figs/%s.indp.png"%kind)
if case == 4:
    pass


# In[ ]:


#from sma import MiddleLatFilter
#start_time = datetime.datetime(2015, 3, 17)
#end_time = datetime.datetime(2015, 3, 17, 10)
#rad, bm = "bks",7

#start_time = datetime.datetime(2017, 4, 4)
#end_time = datetime.datetime(2017, 4, 5)
#rad, bm = "cvw",7

#fdata = FetchData( rad, [start_time, end_time] )
#_, scans = fdata.fetch_data(by="scan", scan_prop={"dur": 2, "stype": "themis"})
#print(" Total numbe of scans: ", len(scans))
#import pickle
#data_dict = pickle.load(open("../data/bks_2015-03-17_scans.pickle", 'rb'))
#data_dict = _filter_by_time(start_time, end_time, data_dict)

#import os
#os.system("rm figs/bks*")
#df = sma_bbox(scans, sdur=15, idx=None)
#from sma import ScatterTypeDetection
#bm=7
#sd = ScatterTypeDetection(df)
#rti = RangeTimePlot(110, np.unique(np.hstack(data_dict["time"])), "", num_subplots=3)
#rti.addParamPlot(df, bm, "Velocity", vel_max=100, vel_step=20, xlabel="")
#rti.addCluster(df, bm, "SMA", xlabel="")
#rti.addGSIS(sd.run(kind=0, case=0), bm, r"GsI:[Sudden]")
#rti.addGSIS(sd.run(kind=0, case=1), bm, r"GsI:[Blanchard 2006]")
#rti.addGSIS(sd.run(kind=0, case=2), bm, r"GsI:[Blanchard 2009]", xlabel='Time, UT')
#rti.addGSIS(sd.run(kind=0, case=3), bm, r"GsI:[X]", xlabel='Time, UT')
#rti.save("figs/bks_sma_01.png")
#rti.close()


# In[ ]:


run = False
if run:
    from sma import MiddleLatFilter
    start_time = datetime.datetime(2017, 4, 4)
    end_time = datetime.datetime(2017, 4, 5)
    rad, bm = "cvw",7

    fdata = FetchData( rad, [start_time, end_time] )
    _, scans = fdata.fetch_data(by="scan", scan_prop={"dur": 2, "stype": "themis"})
    print(" Total numbe of scans: ", len(scans))
    import pickle
    data_dict = pickle.load(open("../data/cvw_2017-04-04_scans.pickle", 'rb'))
    data_dict = _filter_by_time(start_time, end_time, data_dict)

    import os
    os.system("rm figs/cvw*")
    df = sma_bbox(scans, sdur=30, idx=None, dbeam=None, window=5)
    from sma import ScatterTypeDetection
    rti = RangeTimePlot(110, np.unique(np.hstack(data_dict["time"])), "", num_subplots=4)
    rti.addParamPlot(df, bm, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label='Velocity [m/s]')
    rti.addParamPlot(df, bm, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label='Power [dB]')
    rti.addParamPlot(df, bm, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label='Spec. Width [m/s]')
    rti.addCluster(lower_range(df, -1), bm, "SMA", label_clusters=True, skill=estimate_df_skills(df, df.labels), xlabel='Time, UT')
    rti.save("figs/cvw_07_sma.png")
    rti.close()
    sd = ScatterTypeDetection(df)
    rti = RangeTimePlot(110, np.unique(np.hstack(data_dict["time"])), "", num_subplots=5)
    rti.addCluster(lower_range(df, -1), bm, "SMA", label_clusters=True, skill=estimate_df_skills(df, df.labels))
    rti.addGSIS(sd.run(kind=1, case=0), bm, r"GsI:[Sudden]")
    rti.addGSIS(sd.run(kind=1, case=1), bm, r"GsI:[Blanchard 2006]")
    rti.addGSIS(sd.run(kind=1, case=2), bm, r"GsI:[Blanchard 2009]")
    sd = ScatterTypeDetection(lower_range(df, -1))
    rti.addGSIS(sd.run(kind=1, case=3, mod=True), bm, r"GsI:[Chakraborty]", xlabel='Time, UT')
    rti.save("figs/cvw_07_sma_is.png")
    rti.close()


# In[ ]:


run = False
if run:
    from sma import MiddleLatFilter
    start_time = datetime.datetime(2015, 3, 17)
    end_time = datetime.datetime(2015, 3, 17, 12)
    rad, bm = "bks",15

    fdata = FetchData( rad, [start_time, end_time] )
    _, scans = fdata.fetch_data(by="scan", scan_prop={"dur": 2, "stype": "themis"})
    print(" Total numbe of scans: ", len(scans))
    import pickle
    data_dict = pickle.load(open("../data/bks_2015-03-17_scans.pickle", 'rb'))
    data_dict = _filter_by_time(start_time, end_time, data_dict)

    import os
    os.system("rm figs/bks*")
    df = sma_bbox(scans, sdur=30, idx=None, dbeam=15, window=5)
    from sma import ScatterTypeDetection
    rti = RangeTimePlot(110, np.unique(np.hstack(data_dict["time"])), "", num_subplots=4)
    rti.addParamPlot(df, bm, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label='Velocity [m/s]')
    rti.addParamPlot(df, bm, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label='Power [dB]')
    rti.addParamPlot(df, bm, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label='Spec. Width [m/s]')
    rti.addCluster(lower_range(df, -1), bm, "SMA", label_clusters=True, skill=estimate_df_skills(df, df.labels), xlabel='Time, UT')
    rti.save("figs/bks_07_sma.png")
    rti.close()
    sd = ScatterTypeDetection(df)
    rti = RangeTimePlot(110, np.unique(np.hstack(data_dict["time"])), "", num_subplots=5)
    rti.addCluster(lower_range(df, -1), bm, "SMA", label_clusters=True, skill=estimate_df_skills(df, df.labels))
    rti.addGSIS(sd.run(kind=1, case=0), bm, r"GsI:[Sudden]")
    rti.addGSIS(sd.run(kind=1, case=1), bm, r"GsI:[Blanchard 2006]")
    rti.addGSIS(sd.run(kind=1, case=2), bm, r"GsI:[Blanchard 2009]")
    sd = ScatterTypeDetection(lower_range(df, -1))
    rti.addGSIS(sd.run(kind=1, case=3, mod=False), bm, r"GsI:[Chakraborty]", xlabel='Time, UT')
    rti.save("figs/bks_07_sma_is.png")
    rti.close()

