import pandas as pd
import datetime as dt
import os

a_name = "dbscan"
parameters = ['gate', 'beam', 'vel', 'wid', 'time', 'trad_gsflg', 'pow', 'clust_flg']
isgs = {"thresh":[0.5,0.5], "pth":0.5}
plot_params = ["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"]
plot_beams=[7]
gmm = True
save = True

def create_pickle_files():
    # Code to convert any day / radar to ".pickle" file for processing
    from pickle_creator import to_pickle_files
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    dates, rads = [], []
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        dn = start
        rads.append(rad)
        while dn <= end:
            fname = "../data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
            if not os.path.exists(fname):
                dates.append(dn)
            dn = dn + dt.timedelta(days=1)
    to_pickle_files(dates, rads)
    return

def run_algorithms():
    from statistics import run_algorithm
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    dates, rads = [], []
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        dn = start
        rads.append(rad)
        while dn <= end:
            fname = "../data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
            if os.path.exists(fname):
                dates.append(dn)
            dn = dn + dt.timedelta(days=1)
    
    for date in dates:
        for rad in rads:
            print("Batch mode date, rad: %s, %s"%(date.strftime("%Y-%m-%d"), rad))
            gmm_tag = ""
            if gmm: gmm_tag = ".gmm"
            fname = "../outputs/cluster_tags/{rad}.{a_name}{gmm_tag}.{dn}.csv".format(rad=rad, a_name=a_name,gmm_tag=gmm_tag,
                                                                                      dn=date.strftime("%Y%m%d"))
            if not os.path.exists(fname):
                run_algorithm(rad, date, date+dt.timedelta(days=1), a_name, gmm=gmm, 
                              parameters = parameters, isgs=isgs, plot_beams=plot_beams, 
                              plot_params=plot_params, save=save)
        #break
    return

if __name__ == "__main__":
    method = 2
    if method == 1: create_pickle_files()
    if method == 2: run_algorithms()
    pass