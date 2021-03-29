import pandas as pd
import datetime as dt
import os
import dask
import json
import traceback
import numpy as np

import utils
import plotlib

LFS = "LFS/LFS_clustering_superdarn_data/"
a_name = "gb-dbscan"
parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "pow", "clust_flg"]
isgs = {"thresh":[0.5,0.5], "pth":0.5}
plot_params = ["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"]
plot_beams=[7]
gmm = True
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

def run_algorithms():
    pubfile = utils.get_pubfile()
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

def estimate_skill_stats():
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    for start, end, rad, etype in zip(df.event_start, df.event_end, df.rad, df.event_type):
        dates, rads = [], []
        dn = start
        while dn <= end:
            dates.append(dn)
            rads.append(rad)
            dn = dn + dt.timedelta(days=1)
        plotlib.summary_skill_scores(rads, dates, a_name, gmm, etype)
    return

def plot_scatter_histograms():
    a_config = {}
    with open("config/confs.json", "r") as f:
        a_config = json.loads("".join(f.readlines()))
    gmm_tag = "-gmm" if gmm else ""
    fp_details = a_config[a_name + gmm_tag]
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    I = 0
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        ik = np.mod(I,4)
        if ik==2:
            dates, rads = [], []
            dn = start
            while dn <= end:
                dates.append(dn)
                rads.append(rad)
                dn = dn + dt.timedelta(days=1)
            case = 2
            month = start.strftime("%b") + start.strftime("%Y")
            if gmm: fname = "scatter_hist/sctr_{rad}_{a_name}.gmm_case.{case}_kind.1_{month}.png".format(rad=rad, 
                                                                                                         a_name=a_name, case=case,
                                                                                                         month=month)
            else: fname = "scatter_hist/sctr_{rad}_{a_name}_case.{case}_kind.1_{month}.png".format(rad=rad, 
                                                                                                       a_name=a_name, case=case,
                                                                                                       month=month)
            plotlib.histograms_scatters_from_remote(rads, dates, a_name, gmm=gmm, case=case, png_fname=fname, 
                                                    fp_details_list = fp_details[ik], kind=1, gates=10)
        I += 1
    return

def plot_RTI():
    case, kind = 2, 1
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    from statistics import plot_rti_from_arc
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        dn = start
        while dn <= end:
            plot_rti_from_arc(conn, rad, dn, a_name, gmm=gmm, plot_beams=plot_beams, is_local_remove=False, case=case, kind=kind)
            dn = dn + dt.timedelta(days=1)
    conn.close()
    return

def plot_individual_scatter_histograms():
    a_config = {}
    gmm_tag = "-gmm" if gmm else ""
    with open("config/confs_indp.json", "r") as f:
        a_config = json.loads("".join(f.readlines()))
    fp_details = a_config[a_name + gmm_tag]
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    I = 0
    for start, end, rad in zip(df.event_start, df.event_end, df.rad):
        ik = np.mod(I,4)
        if ik>=0:
            dates, rads = [], []
            dn = start
            while dn <= end:
                dates.append(dn)
                rads.append(rad)
                dn = dn + dt.timedelta(days=1)
            case = 2
            month = start.strftime("%b") + start.strftime("%Y")
            if gmm: fname = "scatter_hist_indp/sctr_{rad}_{a_name}.gmm_case.{case}_kind.0_{month}.png".format(rad=rad, 
                                                                                                 a_name=a_name, case=case, month=month)
            else: fname = "scatter_hist_indp/sctr_{rad}_{a_name}_case.{case}_kind.0_{month}.png".format(rad=rad, 
                                                                                                        a_name=a_name, case=case, 
                                                                                                        month=month)
            plotlib.histograms_indp_scatters_from_remote(rads, dates, a_name, gmm=gmm, case=case, png_fname=fname, 
                                                    fp_details_list = fp_details[ik], kind=0)
        I += 1
    return

def plot_2D_histograms():
    gmm_tag = "-gmm" if gmm else ""
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    I = 0
    for start, end, rad, e in zip(df.event_start, df.event_end, df.rad, df.event_type):
        dates, rads = [], []
        dn = start
        while dn <= end:
            dates.append(dn)
            rads.append(rad)
            dn = dn + dt.timedelta(days=1)
        png_fname = "hist_2D_%s_%s.png"%(rad,e)
        plotlib.plot_2D_hist(rads, dates, a_name, gmm=gmm, case=2, kind=0, png_fname=png_fname)
        I += 1
    return

def plot_RTI_riberio():
    from statistics import ribiero_gflg_RTI
    gmm_tag = "-gmm" if gmm else ""
    df = pd.read_csv("events.txt", parse_dates=["event_start", "event_end"])
    I = 0
    for start, end, rad, e in zip(df.event_start, df.event_end, df.rad, df.event_type):
        dates, rads = [], []
        dn = start
        while dn <= end:
            dates.append(dn)
            rads.append(rad)
            dn = dn + dt.timedelta(days=1)
        ribiero_gflg_RTI(rads, dates, a_name, gmm=gmm, case=2, kind=0)
        I += 1
        break
    return

if __name__ == "__main__":
    method = 8
    if method == 1: create_pickle_files()
    if method == 2: run_algorithms()
    if method == 3: estimate_skill_stats()
    if method == 4: plot_scatter_histograms()
    if method == 5: plot_RTI()
    if method == 6: plot_individual_scatter_histograms()
    if method == 7: plot_2D_histograms()
    if method == 8: plot_RTI_riberio()
    pass
