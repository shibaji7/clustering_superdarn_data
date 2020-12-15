import numpy as np
import sys
sys.path.insert(0, "..")
from algorithms.dbscan_gmm import DBSCAN_GMM
from algorithms.grid_based_dbscan import GridBasedDBSCAN
from algorithms.grid_based_dbscan_gmm import GridBasedDBSCAN_GMM

import pandas as pd
from plotlib import RangeTimePlot
import os

from utils import Skills, ScatterTypeDetection

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

def estimate_df_skills(df, skill_file, save):
    V, W, L = np.array(df.v), np.array(df.w_l), np.array(df.labels)
    X = np.array([V.tolist(), W.tolist()]).T
    X[np.isnan(X)] = 0.
    X[np.isinf(X)] = 0.
    sk = Skills(X, L)
    if save:
        with open(skill_file, "w") as f:
            f.write("chscore,bhscore,hscore,xuscore\n")
            f.write(str(sk.chscore)+","+str(sk.bhscore)+","+str(sk.hscore)+","+str(sk.xuscore)+"\n")
    return sk

def save_tags_stats(clusters, df, rad, a_name, dn):
    dat = []
    for case in clusters.keys():
        for bm in clusters[case].keys():
            for cls in clusters[case][bm].keys():
                x = clusters[case][bm][cls]
                x["idx"] = "%d_%d_%d"%(case,bm,cls)
                dat.append(x)
    d = pd.DataFrame.from_records(dat)
    eff_file = "../outputs/efficiency/{rad}.{a_name}.{dn}.csv".format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
    d.to_csv(eff_file, index=False, header=True)
    clust_file = "../outputs/cluster_tags/{rad}.{a_name}.{dn}.csv".format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
    df.to_csv(clust_file, index=False, header=True)
    return

def run_algorithm(rad, start, end, a_name="dbscan", gmm=False, 
                  parameters = ["gate", "beam", "vel", "wid", "time", "trad_gsflg", "elv", "pow", "clust_flg"],
                  isgs={"thresh":[0.5,0.5], "pth":0.5}, plot_beams=[7], 
                  plot_params=["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"], save=True):
    
    if a_name=="dbscan": algo = DBSCAN_GMM(start, end, rad, BoxCox=True, load_model=False, save_model=False, run_gmm=gmm)
    if a_name=="gb-dbscan" and np.logical_not(gmm): algo = GridBasedDBSCAN(start_time, end_time, rad, load_model=False, save_model=False)
    if a_name=="gb-dbscan" and gmm: algo = GridBasedDBSCAN_GMM(start_time, end_time, rad, load_model=False, save_model=False,
                features=["beam", "gate", "time","vel","wid"], scan_eps=1)
    df = to_pandas(algo.data_dict, keys=parameters)
    skill_file = "../outputs/skills/{rad}.{a_name}.{dn}.csv".format(rad=rad, a_name=a_name, dn=start.strftime("%Y%m%d"))
    skills = estimate_df_skills(df, skill_file, save)
    std = ScatterTypeDetection(df.copy())
    df, clusters = std.run(thresh=isgs["thresh"], pth=isgs["pth"])
    for beam in plot_beams:
        fig_title = "Rad:{rad}, Model:{a_name}, Beam:{bm}, Date:{dn} UT".format(rad=rad, a_name=a_name, bm="%02d"%beam,
                                                                                dn=start.strftime("%Y-%m-%d"))
        os.system("mkdir -p ../outputs/figures/{rad}/{dn}/".format(rad=rad, dn=start.strftime("%Y-%m-%d")))
        fname = "../outputs/figures/{rad}/{dn}/{bm}.{a_name}.png".format(rad=rad, dn=start.strftime("%Y-%m-%d"), 
                                                                       bm="%02d"%beam, a_name=a_name)
        rti = RangeTimePlot(100, np.unique(df.time), fig_title, num_subplots=6)
        rti.addParamPlot(df, beam, "Velocity", p_max=100, p_min=-100, p_step=25, xlabel="", zparam="v", label="Velocity [m/s]")
        rti.addParamPlot(df, beam, "Power", p_max=30, p_min=3, p_step=3, xlabel="", zparam="p_l", label="Power [dB]")
        rti.addParamPlot(df, beam, "Spec. Width", p_max=100, p_min=0, p_step=10, xlabel="", zparam="w_l", label="Spec. Width [m/s]")
        rti.addCluster(df, beam, a_name, label_clusters=True, skill=skills, xlabel="")
        rti.addGSIS(df, beam, GS_CASES[0], xlabel="", zparam="gflg_0_0")
        rti.addGSIS(df, beam, GS_CASES[0], xlabel="Time, UT", zparam="gflg_0_1", clusters=clusters[0], label_clusters=True)
        rti.save(fname)
        rti.close()
        pass
    if save: save_tags_stats(clusters, df, rad, a_name, start)
    os.system("rm ../data/{rad}_{dn}_scans_db.csv".format(rad=rad, dn=start.strftime("%Y-%m-%d")))
    return