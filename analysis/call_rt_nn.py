import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
import pickle

import datetime as dt
import os
import time
import bz2
import glob

import dask

import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter, num2date

import pydarn
import pydarnio
import sys
sys.path.append("rt_nn_model")
import predict_sct_type

import utils
import sys
sys.path.insert(0, "..")
from algorithms.dbscan_gmm import DBSCAN_GMM
from plotlib import RangeTimePlot


LFS = "LFS/LFS_clustering_superdarn_data/"

def to_pandas(dicts, keys):
    df = pd.DataFrame()
    _o = {}
    for k in keys:
        _o[k] = []
        for x in dicts[k]:
            _o[k].extend(x)
    df = pd.DataFrame.from_records(_o)
    df = df.rename(columns={"gate":"slist", "beam":"bmnum", "vel":"v", "wid":"w_l", 
                            "time":"time", "pow":"p_l", "clust_flg":"labels"})
    return df


def get_algo_name_remove_file(a_name, gmm):
    algo_map = {"dbscan": ["db", "dbgmm"], "gb-dbscan":["gb-dbscan", "gb-dbscan.gmm"]}
    if gmm: uname = algo_map[a_name][1]
    else: uname = algo_map[a_name][0]
    return uname

def save_tags_stats(conn, df, rad, a_name, dn, gmm_tag, rnn):
    if rnn: clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}_proc.csv".format(rad=rad, a_name=a_name, 
                                                                                   gt=gmm_tag, dn=dn.strftime("%Y%m%d"))
    else: clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, 
                                                                                   gt=gmm_tag, dn=dn.strftime("%Y%m%d"))
    df.to_csv(clust_file, index=False, header=True)
    utils.to_remote_FS(conn, clust_file, LFS, is_local_remove=True)
    return

def create_data_frame(date_arr, freq_arr, azm_arr, gate_arr, beam_arr, vel_arr, gflag_arr):
    fit_data_df = pd.DataFrame(
            {"date": date_arr,
             "freq": freq_arr,
             "azmth": azm_arr,
             "gate": gate_arr,
             "beam": beam_arr,
             "vel": vel_arr,
             "gflag": gflag_arr
            })
    return fit_data_df

def predict_rt_nn(fit_data_df, date_range, model_name = "rt_nn_model/weights.epoch_08.val_loss_0.13.hdf5", 
                  scaler_name = "rt_nn_model/scaler.pck", inds_data_name = "rt_nn_model/inds_data.pkl"):
    fit_data_df = fit_data_df.set_index("date")
    
    inds_data = pd.read_pickle(inds_data_name)
    inds_data = inds_data[(inds_data["date"] >= date_range[0]) & (inds_data["date"] <= date_range[1] + dt.timedelta(days=1))]
    inds_data = inds_data.set_index("date")
    inds_data = inds_data.reindex(fit_data_df.index, method='nearest')
    
    fit_data_df = fit_data_df.join(inds_data)
    fit_data_df.sort_index(inplace=True)
    fit_data_df.reset_index(inplace=True)
    fit_data_df.drop_duplicates(inplace=True)
    pred_obj = predict_sct_type.PredctSctr( model_name, scaler_name, fit_data_df)
    pred_data_df = pred_obj.predict(batch_size=512)
    return pred_data_df

def create_pickle_files(date_range, rad):
    # Code to convert any day / radar to ".pickle" file for processing
    dates = []
    radar, dates, dn, end = "bks", [], date_range[0], date_range[1]
    while dn <= end:
        dates.append(dn)
        dn += dt.timedelta(days=1)
        
    from pickle_creator import to_pickle_files
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    dask_out = []
    lfname = []
    for dn in dates:
        fname = LFS + "data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
        if not utils.chek_remote_file_exists(fname, conn):
            dask_out.append(dask.delayed(to_pickle_files)([dn], [rad], conn, LFS, is_local_remove=True))
    _ = [do.compute() for do in dask_out]
    conn.close()
    return

def pre_processing(date_range, rad, model_name = "rt_nn_model/weights.epoch_08.val_loss_0.13.hdf5",
                  scaler_name = "rt_nn_model/scaler.pck", inds_data_name = "rt_nn_model/inds_data.pkl"):
    from pickle_creator import to_pkl_file
    dates = []
    radar, dates, dn, end = "bks", [], date_range[0], date_range[1]
    while dn <= end:
        dates.append(dn)
        dn += dt.timedelta(days=1)
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    for dn in dates:
        to_file = LFS + "data/%s_%s_scans_proc.pickle"%(rad, dn.strftime("%Y-%m-%d"))
        if not utils.chek_remote_file_exists(to_file, conn):
            local_file = "../data/%s_%s_scans.pickle"%(rad, dn.strftime("%Y-%m-%d"))
            utils.from_remote_FS(conn, local_file, LFS)
            csv = pd.read_pickle(local_file)
            gate, beam, vel, wid, time, trad_gsflg, elv, powr, nrang, nbeam, tfreq, bmazm =\
                                [], [], [], [], [], [], [], [], [], [], [], []
            for _i in range(len(csv["time"])):
                L = len(csv["gate"][_i])
                gate.extend(csv["gate"][_i])
                vel.extend(csv["vel"][_i])
                wid.extend(csv["wid"][_i])
                trad_gsflg.extend(csv["trad_gsflg"][_i])
                elv.extend(csv["elv"][_i])
                powr.extend(csv["pow"][_i])
                beam.extend(csv["beam"][_i])
                time.extend(num2date(csv["time"][_i]))
                tfreq.extend(csv["tfreq"][_i])
                bmazm.extend(csv["bmazm"][_i])
                nrang.extend([csv["nrang"]]*L)
                nbeam.extend([csv["nbeam"]]*L)
                if len(csv["elv"][_i]) != L: elv.extend([None]*(L-len(csv["elv"][_i])))
            if len(elv) == len(gate): df = pd.DataFrame.from_dict({"gate":gate, "vel":vel, "wid":wid, "elv":elv,
                                                                   "trad_gsflg":trad_gsflg, "pow":powr, "beam":beam,
                                                                   "time":time, "nrang":nrang, "nbeam":nbeam,
                                                                   "bmazm":bmazm, "tfreq":tfreq})
            else: df = pd.DataFrame.from_dict({"gate":gate, "vel":vel, "wid":wid,
                                               "trad_gsflg":trad_gsflg, "pow":powr, "beam":beam,
                                               "time":time, "nrang":nrang, "nbeam":nbeam,
                                               "bmazm":bmazm, "tfreq":tfreq})
            df.time = df["time"].dt.tz_localize(None)
            fit_df = create_data_frame(df.time, df.tfreq, df.bmazm, df.gate, df.beam, df.vel, df.trad_gsflg)
            pred_df = predict_rt_nn(fit_df, date_range, model_name, scaler_name, inds_data_name)
            df["hf_label"] = np.copy(pred_df["iono_hf_sct_label_pred"])
            df["onehf_label"] = np.copy(pred_df["iono_one_hf_sct_label_pred"])
            df["one_label"] = np.copy(pred_df["gnd_one_sct_label_pred"])
            df["two_label"] = np.copy(pred_df["gnd_two_sct_label_pred"])
            df["no_label"] = np.copy(pred_df["no_sct_label_pred"])
            #print(df.tail(50))
            to_pkl_file(df, dn, rad, conn, LFS, is_local_remove=True)
    conn.close()
    return

def run_algorithm(date_range, rad, a_name="dbscan", gmm=True, save=True, rnn=True):
    if rnn: parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "elv", "pow", "clust_flg", "hf_label", 
                  "onehf_label", "one_label", "two_label", "no_label"]
    else: parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "elv", "pow", "clust_flg"]
    dates = []
    radar, dates, dn, end = "bks", [], date_range[0], date_range[1]
    while dn <= end:
        dates.append(dn)
        dn += dt.timedelta(days=1)
    gmm_tag = ""
    if gmm: gmm_tag = ".gmm"
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    if rnn: features = ["gate", "beam", "vel", "wid", "time", "hf_label", "onehf_label", "one_label", "two_label", "no_label"]
    else: features = ["gate", "beam", "vel", "wid", "time"]
    for dn in dates:
        local_file = "../data/%s_%s_scans_proc.pickle"%(rad, dn.strftime("%Y-%m-%d"))
        utils.from_remote_FS(conn, local_file, LFS)
        if a_name=="dbscan": algo = DBSCAN_GMM(dn, dn + dt.timedelta(days=1), rad, BoxCox=False, rnn=rnn,
                                               load_model=False, save_model=False, run_gmm=gmm, features=features)
        df = to_pandas(algo.data_dict, keys=parameters)
        df = utils._run_riberio_threshold_on_rad(df)
        if save: save_tags_stats(conn, df, rad, a_name, dn, gmm_tag, rnn)
        if os.path.exists(local_file): os.remove(local_file)
    os.system("rm -rf ../data/*")
    os.system("rm -rf ../outputs/figures/*")
    return

def plot_RTI_images(date_range, rad, a_name="dbscan", gmm=True, rnn=True):
    dates = []
    radar, dates, dn, end = "bks", [], date_range[0], date_range[1]
    while dn <= end:
        dates.append(dn)
        dn += dt.timedelta(days=1)
    gmm_tag = ""
    if gmm: gmm_tag = ".gmm"
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    for dn in dates:
        if rnn: clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}_proc.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, 
                                                                              dn=dn.strftime("%Y%m%d"))
        else: clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, 
                                                                              dn=dn.strftime("%Y%m%d"))
        utils.fetch_file(conn, clust_file, LFS)
        if os.path.exists(clust_file):
            df = pd.read_csv(clust_file)
            df.hf_label = df.hf_label*100
            df.onehf_label = df.onehf_label*100
            df.one_label = df.one_label*100
            df.two_label = df.two_label*100
            df.no_label = df.no_label*100
            for beam in np.unique(df.bmnum):
                fig_title = "Rad:{rad}, Model:{a_name}, Beam:{bm}, Date:{dn} UT".format(rad=rad, a_name=a_name, bm="%02d"%beam,
                                                                                        dn=dn.strftime("%Y-%m-%d"))
                ldir = "../outputs/figures/{rad}/{dn}/".format(rad=rad, dn=dn.strftime("%Y-%m-%d"))
                os.system("mkdir -p " + ldir)
                if rnn: fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}{gt}_proc.png".format(rad=rad, dn=dn.strftime("%Y-%m-%d"), 
                                                                               bm="%02d"%beam, a_name=a_name, gt=gmm_tag)
                else: fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}{gt}.png".format(rad=rad, dn=dn.strftime("%Y-%m-%d"), 
                                                                               bm="%02d"%beam, a_name=a_name, gt=gmm_tag)
                rti = RangeTimePlot(100, np.unique(df.time), fig_title, num_subplots=10)
                rti.addParamPlot(df, beam, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]")
                rti.addParamPlot(df, beam, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label="Power [dB]")
                rti.addParamPlot(df, beam, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spec. Width [m/s]")
                rti.addParamPlot(df, beam, "0.5 Hops", p_max=100, p_min=0, p_step=10, xlabel="", zparam="hf_label", label="Probs. [%]")
                rti.addParamPlot(df, beam, "1.5 Hops", p_max=100, p_min=0, p_step=10, xlabel="", zparam="onehf_label", label="Probs. [%]")
                rti.addParamPlot(df, beam, "1.0 Hops", p_max=100, p_min=0, p_step=10, xlabel="", zparam="one_label", label="Probs. [%]")
                rti.addParamPlot(df, beam, "2.0 Hops", p_max=100, p_min=0, p_step=10, xlabel="", zparam="two_label", label="Probs. [%]")
                rti.addParamPlot(df, beam, "0.0 Hops", p_max=100, p_min=0, p_step=10, xlabel="", zparam="no_label", label="Probs. [%]")
                rti.addCluster(df, beam, a_name, label_clusters=True, skill=None, xlabel="")
                rti.save(fname)
                rti.close()
    return

if __name__ == "__main__":
    rnn = True
    #create_pickle_files([dt.datetime(2010,1,15), dt.datetime(2010,1,15)], "bks")
    #pre_processing([dt.datetime(2010,1,15), dt.datetime(2010,1,15)], "bks")
    #run_algorithm([dt.datetime(2010,1,15), dt.datetime(2010,1,15)], "bks", rnn=rnn)
    #plot_RTI_images([dt.datetime(2010,1,15), dt.datetime(2010,1,15)], "bks", rnn=rnn)