import numpy as np
import sys
sys.path.insert(0, "..")
from algorithms.dbscan_gmm import DBSCAN_GMM
from algorithms.grid_based_dbscan import GridBasedDBSCAN
from algorithms.grid_based_dbscan_gmm import GridBasedDBSCAN_GMM

import pandas as pd
from plotlib import RangeTimePlot
import os

import utils
from utils import Skills, ScatterTypeDetection

LFS = "LFS/LFS_clustering_superdarn_data/"
GS_CASES = ["Sudden [2004]", "Blanchard [2006]", "Blanchard [2009]"]
def to_pandas(dicts, keys=["gate", "beam", "vel", "wid", "time", "trad_gsflg", "elv", "pow", "clust_flg"]):
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

def estimate_df_skills(conn, df, skill_file, save):
    V, W, L = np.array(df.v), np.array(df.w_l), np.array(df.labels)
    X = np.array([V.tolist(), W.tolist()]).T
    X[np.isnan(X)] = 0.
    X[np.isinf(X)] = 0.
    sk = Skills(X, L)
    if save:
        with open(skill_file, "w") as f:
            f.write("chscore,bhscore,hscore,xuscore,mdbhscore,mdhscore,mdxuscore\n")
            f.write(str(sk.chscore)+","+str(sk.bhscore)+","+str(sk.hscore)+","+str(sk.xuscore)+","+\
                    str(sk.mdbhscore)+","+str(sk.mdhscore)+","+str(sk.mdxuscore)+"\n")
        utils.to_remote_FS(conn, skill_file, LFS, is_local_remove=True)
    return sk

def save_tags_stats(conn, clusters, df, rad, a_name, dn, gmm_tag):
    dat = []
    if clusters:
        for case in clusters.keys():
            for bm in clusters[case].keys():
                for cls in clusters[case][bm].keys():
                    x = clusters[case][bm][cls]
                    x["idx"] = "%d_%d_%d"%(case,bm,cls)
                    dat.append(x)
        d = pd.DataFrame.from_records(dat)
        eff_file = "../outputs/efficiency/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, dn=dn.strftime("%Y%m%d"))
        d.to_csv(eff_file, index=False, header=True)
        utils.to_remote_FS(conn, eff_file, LFS, is_local_remove=True)
    clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, dn=dn.strftime("%Y%m%d"))
    df.to_csv(clust_file, index=False, header=True)
    utils.to_remote_FS(conn, clust_file, LFS, is_local_remove=True)
    return

def get_algo_name_remove_file(a_name, gmm):
    algo_map = {"dbscan": ["db", "dbgmm"], "gb-dbscan":["gb-dbscan", "gb-dbscan.gmm"]}
    if gmm: uname = algo_map[a_name][1]
    else: uname = algo_map[a_name][0]
    return uname

def run_algorithm(conn, rad, start, end, a_name="dbscan", gmm=False, 
                  parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "elv", "pow", "clust_flg"],
                  isgs={"thresh":[0.5,0.5], "pth":0.5}, plot_beams=[7], 
                  plot_params=["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"], save=True):
    gmm_tag = ""
    if gmm: gmm_tag = ".gmm"
    close = False
    if conn==None: 
        close = True
        pubfile = utils.get_pubfile()
        conn = utils.get_session(key_filename=pubfile)
    local_file = "../data/%s_%s_scans.pickle"%(rad, start.strftime("%Y-%m-%d"))
    utils.from_remote_FS(conn, local_file, LFS)
    if a_name=="dbscan": algo = DBSCAN_GMM(start, end, rad, BoxCox=True, load_model=False, save_model=False, run_gmm=gmm)
    if a_name=="gb-dbscan" and np.logical_not(gmm): algo = GridBasedDBSCAN(start, end, rad, load_model=False, save_model=False)
    if a_name=="gb-dbscan" and gmm: algo = GridBasedDBSCAN_GMM(start, end, rad, load_model=False, save_model=False,
                features=["beam", "gate", "time","vel","wid"], scan_eps=1)
    print(algo.data_dict.keys())
    df = to_pandas(algo.data_dict, keys=parameters)
    skill_file = "../outputs/skills/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, dn=start.strftime("%Y%m%d"))
    #skills = estimate_df_skills(conn, df, skill_file, save)
    clusters = None
    #std = ScatterTypeDetection(df.copy())
    #df, clusters = std.run(thresh=isgs["thresh"], pth=isgs["pth"])
    for beam in plot_beams:
        try:
            fig_title = "Rad:{rad}, Model:{a_name}, Beam:{bm}, Date:{dn} UT".format(rad=rad, a_name=a_name, bm="%02d"%beam,
                                                                                dn=start.strftime("%Y-%m-%d"))
            ldir = "../outputs/figures/{rad}/{dn}/".format(rad=rad, dn=start.strftime("%Y-%m-%d"))
            os.system("mkdir -p " + ldir)
            fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}{gt}.png".format(rad=rad, dn=start.strftime("%Y-%m-%d"), 
                                                                       bm="%02d"%beam, a_name=a_name, gt=gmm_tag)
            rti = RangeTimePlot(100, np.unique(df.time), fig_title, num_subplots=6)
            rti.addParamPlot(df, beam, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]")
            rti.addParamPlot(df, beam, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label="Power [dB]")
            rti.addParamPlot(df, beam, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spec. Width [m/s]")
            rti.addCluster(df, beam, a_name, label_clusters=True, skill=skills, xlabel="")
            rti.addGSIS(df, beam, GS_CASES[0], xlabel="", zparam="gflg_0_0")
            rti.addGSIS(df, beam, GS_CASES[0], xlabel="Time, UT", zparam="gflg_0_1", clusters=clusters[0], label_clusters=True)
            rti.save(fname)
            rti.close()
            utils.to_remote_FS_dir(conn, ldir, fname, LFS, is_local_remove=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("Exception in printing beam:", beam)
        pass
    df = utils._run_riberio_threshold_on_rad(df)
    if save: save_tags_stats(conn, clusters, df, rad, a_name, start, gmm_tag)
    if os.path.exists(local_file): os.remove(local_file)
    cache = "../data/{rad}_{dn}_scans_{algo}.csv".format(rad=rad, dn=start.strftime("%Y-%m-%d"), 
                                                         algo=get_algo_name_remove_file(a_name, gmm))
    if os.path.exists(cache): os.remove(cache)
    os.system("rm -rf ../outputs/figures/*")
    if close: conn.close()
    return

def plot_rti_from_arc(conn, rad, date, a_name, gmm=False, plot_beams=[7], is_local_remove=False, 
                      isgs={"thresh":[0.5,0.5], "pth":0.5}, cache_clean=True, case=2, kind=1):
    gmm_tag = ".gmm" if gmm else ""
    clust_file = "../outputs/cluster_tags/{rad}.{a_name}{gt}.{dn}.csv".format(rad=rad, a_name=a_name, gt=gmm_tag, 
                                                                              dn=date.strftime("%Y%m%d"))
    utils.fetch_file(conn, clust_file, LFS)
    if os.path.exists(clust_file):
        df = pd.read_csv(clust_file)
        std = ScatterTypeDetection(df.copy())
        df, clusters = std.run(thresh=isgs["thresh"], pth=isgs["pth"])
        for beam in plot_beams:
            fig_title = "Rad:{rad}, Model:{a_name}, Beam:{bm}, Date:{dn} UT".format(rad=rad, a_name=a_name, bm="%02d"%beam,
                                                                                    dn=date.strftime("%Y-%m-%d"))
            ldir = "../outputs/figures/{rad}/{dn}/".format(rad=rad, dn=date.strftime("%Y-%m-%d"))
            os.system("mkdir -p " + ldir)
            fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}{gt}.png".format(rad=rad, dn=date.strftime("%Y-%m-%d"), 
                                                                           bm="%02d"%beam, a_name=a_name, gt=gmm_tag)
            print(df.head())
            rti = RangeTimePlot(100, np.unique(df.time), fig_title, num_subplots=6)
            rti.addParamPlot(df, beam, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]")
            rti.addParamPlot(df, beam, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label="Power [dB]")
            rti.addParamPlot(df, beam, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spec. Width [m/s]")
            rti.addCluster(df, beam, a_name, label_clusters=True, skill=None, xlabel="")
            rti.addGSIS(df, beam, GS_CASES[case], xlabel="", zparam="gflg_%d_0"%case)
            rti.addGSIS(df, beam, GS_CASES[case], xlabel="Time, UT", zparam="gflg_%d_%d"%(case, kind), 
                        clusters=clusters[case], label_clusters=True)
            rti.save(fname)
            rti.close()
            utils.to_remote_FS_dir(conn, ldir, fname, LFS, is_local_remove=is_local_remove)
        
    if cache_clean: os.system("rm -rf ../outputs/figures/*")
    #os.system("rm -rf ../outputs/cluster_tags/*")
    return

def ribiero_gflg_RTI(rads, dates, a_name, gmm, case, kind):
    gmm_tag = ".gmm" if gmm else ""
    pubfile = utils.get_pubfile()
    conn = utils.get_session(key_filename=pubfile)
    if gmm: rfname = "../outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    else: rfname = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv"
    X, ix = pd.DataFrame(), 0
    for rad, dn in zip(rads, dates):
        floc = rfname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
        print("File - ", floc)
        utils.fetch_file(conn, floc, LFS)
        if os.path.exists(floc):
            u = pd.read_csv(floc)
            clust_type = "gflg_%d_%d"%(case,kind)
            for beam in np.unique(u.bmnum):
                try:
                    df = utils._run_riberio_threshold(u, beam)
                    fig_title = "Rad:{rad}, Model:{a_name}, Beam:{bm}, Date:{dn} UT".format(rad=rad, a_name=a_name, bm="%02d"%beam,
                                                                                        dn=dn.strftime("%Y-%m-%d"))
                    ldir = "../outputs/figures/{rad}/{dn}/".format(rad=rad, dn=dn.strftime("%Y-%m-%d"))
                    os.system("mkdir -p " + ldir)
                    fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}{gt}.png".format(rad=rad, dn=dn.strftime("%Y-%m-%d"), 
                                                                               bm="%02d"%beam, a_name=a_name, gt=gmm_tag)
                    rti = RangeTimePlot(100, np.unique(df.time), fig_title, num_subplots=6)
                    rti.addParamPlot(df, beam, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]")
                    rti.addParamPlot(df, beam, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label="Power [dB]")
                    rti.addParamPlot(df, beam, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spec. Width [m/s]")
                    rti.addCluster(df, beam, a_name, label_clusters=True, skill=None, xlabel="")
                    rti.addGSIS(df, beam, GS_CASES[case], xlabel="", zparam="gflg_%d_0"%case)
                    rti.addGSIS(df, beam, "Ribiero", xlabel="Time, UT", zparam="ribiero_gflg")
                    rti.save(fname)
                    rti.close()
                except Exception: 
                    import traceback
                    traceback.print_exc()
                    print("Exception in printing beam:", beam)
        ix += 1
    #os.system("rm -rf ../outputs/cluster_tags/*")
    return