import pandas as pd
import datetime as dt
import os
import dask
import utils
import json
import traceback

LFS = "LFS/LFS_clustering_superdarn_data/"
a_name = "gb-dbscan"
parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "pow", "clust_flg"]
isgs = {"thresh":[0.5,0.5], "pth":0.5}
plot_params = ["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"]
plot_beams=[7]
gmm = False
save = True

def create_pickle_files():
    # Code to convert any day / radar to ".pickle" file for processing
    from pickle_creator import to_pickle_files
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    dask_out = []
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        dn = start
        while dn <= end:
            fname = "../data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
            if not os.path.exists(fname):
                dask_out.append(dask.delayed(to_pickle_files)([dn], [rad]))
            dn = dn + dt.timedelta(days=1)
    _ = [do.compute() for do in dask_out]
    return

def get_pubfile():
    with open("pub.json", "r") as f:
        obj = json.loads("".join(f.readlines()))
        pubfile = obj["pubfile"]
    return pubfile

def run_algorithms():
    pubfile = get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    from statistics import run_algorithm
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    dates, rads = [], []
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        dn = start
        while dn <= end:
            fname = LFS + "data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
            if utils.chek_remote_file_exists(fname, conn):
                dates.append(dn)
                rads.append(rad)
            dn = dn + dt.timedelta(days=1)
    
    for date, rad in zip(dates, rads):
        print("Batch mode date, rad: %s, %s"%(date.strftime("%Y-%m-%d"), rad))
        gmm_tag = ""
        if gmm: gmm_tag = ".gmm"
        fname = LFS + "outputs/cluster_tags/{rad}.{a_name}{gmm_tag}.{dn}.csv".format(rad=rad, a_name=a_name,gmm_tag=gmm_tag,
                                                                                  dn=date.strftime("%Y%m%d"))
        if not utils.chek_remote_file_exists(fname, conn):
            try:
                run_algorithm(conn, rad, date, date+dt.timedelta(days=1), a_name, gmm=gmm, 
                              parameters = parameters, isgs=isgs, plot_beams=plot_beams, 
                              plot_params=plot_params, save=save)
            except:
                print(" Error running algo - ", a_name,gmm_tag, rad, dn)
                traceback.print_exc()
            #break
    conn.close()
    return

if __name__ == "__main__":
    method = 2
    if method == 1: create_pickle_files()
    if method == 2: run_algorithms()
    pass
