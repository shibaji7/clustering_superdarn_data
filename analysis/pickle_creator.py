import pickle
from matplotlib.dates import date2num, num2date
import datetime as dt
import pandas as pd
import pdb
from get_sd_data import *

import traceback

import utils
def to_db(date, rad, conn, LFS, is_local_remove=True):
    fdata = FetchData( rad, [date,
                date + dt.timedelta(days=1)] )
    _, scans = fdata.fetch_data(by="scan")
    #recs = fdata.convert_to_pandas(_beams_)
    print(" Beams per scan:", len(scans[int(len(scans)/2)].beams))
    print(" Total scans:", len(scans))

    gate_scans = []
    beam_scans = []
    vel_scans = []
    wid_scans = []
    time_scans = []
    trad_gs_flg_scans = []
    elv_scans = []
    power_scans = []
    
    time = []
    beam = []
    freq = []
    nsky = []
    nsch = []
    bazm = []
    
    bmax, nrang = 0, 0
    for i in range(len(scans)):
        s = scans[i]
        g, bm, v, w, el, tr, tm, n, ns, f, pw, bzm = [], [], [], [], [], [], [], [], [], [], [], []
        for b in s.beams:
            g.extend(np.array(b.slist).tolist())
            bm.extend([b.bmnum]*len(b.slist))
            v.extend(np.array(b.v).tolist())
            w.extend(np.array(b.w_l).tolist())
            tr.extend(np.array(b.gflg).tolist())
            el.extend(np.array(b.elv).tolist())
            tm.extend([date2num(b.time)]*len(b.slist))
            pw.extend(np.array(b.p_l).tolist())
            f.extend([b.tfreq/1e3]*len(b.slist))
            bzm.extend([b.bmazm]*len(b.slist))
        if i==0: bmax, nrang = np.max(bm), b.nrang
        gate_scans.append(np.array(g))
        beam_scans.append(np.array(bm))
        vel_scans.append(np.array(v))
        wid_scans.append(np.array(w))
        time_scans.append(np.array(tm))
        trad_gs_flg_scans.append(np.array(tr))
        elv_scans.append(np.array(el))
        power_scans.append(np.array(pw))
        bazm.append(np.array(bzm))
        freq.append(np.array(f))
        
    data = {'gate' : gate_scans, 'beam' : beam_scans, 'vel' : vel_scans, 'wid': wid_scans,
            'time' : time_scans, 'trad_gsflg' : trad_gs_flg_scans, 'elv': elv_scans, "pow":power_scans,
            'nrang' : nrang, 'nbeam' : bmax + 1, 'bmazm':bazm, 'tfreq':freq}
    filename = "../data/%s_%s_scans" % (rad, date.strftime("%Y-%m-%d")) 
    pickle.dump(data, open(filename+".pickle", 'wb'))
    utils.to_remote_FS(conn, filename+".pickle", LFS, is_local_remove)
    return

def to_pkl_file(df, date, rad, conn, LFS, is_local_remove=True):
    
    def check_nans(x):
        e = np.copy(x)
        if np.isnan(e).any(): e = e[~np.isnan(e)]
        return e
    
    gate_scans = []
    beam_scans = []
    vel_scans = []
    wid_scans = []
    time_scans = []
    trad_gs_flg_scans = []
    elv_scans = []
    power_scans = []
    tfreq_scans = []
    bazm_scans = []
    
    hf_label = []
    onehf_label = []
    one_label = []
    two_label = []
    no_label = []
    
    bmax, nrang = df.nbeam.tolist()[0], df.nrang.tolist()[0]
    gb = df.groupby("time")
    for index, row in gb:
        L = len(row)
        gate_scans.append(np.array(row["gate"]))
        beam_scans.append(np.array(row["beam"]))
        vel_scans.append(np.array(row["vel"]))
        wid_scans.append(np.array(row["wid"]))
        trad_gs_flg_scans.append(np.array(row["trad_gsflg"]))
        #elv_scans.append(check_nans(np.array(row["elv"])))
        elv_scans.append(np.array(row["elv"]))
        power_scans.append(np.array(row["pow"]))
        bazm_scans.append(np.array(row["bmazm"]))
        tfreq_scans.append(np.array(row["tfreq"]))
        time_scans.append([date2num(index)]*L)
        hf_label.append(np.array(row["hf_label"]))
        onehf_label.append(np.array(row["onehf_label"]))
        one_label.append(np.array(row["one_label"]))
        two_label.append(np.array(row["two_label"]))
        no_label.append(np.array(row["no_label"]))
    data = {'gate' : gate_scans, 'beam' : beam_scans, 'vel' : vel_scans, 'wid': wid_scans,
            'time' : time_scans, 'trad_gsflg' : trad_gs_flg_scans, 'elv': elv_scans, 'pow':power_scans,
            'nrang' : nrang, 'nbeam' : bmax, 'bmazm':bazm_scans, 'tfreq':tfreq_scans, 
            'hf_label' : hf_label, 'onehf_label' : onehf_label, 'one_label' : one_label, 'two_label' : two_label,
            'no_label' : no_label }
    filename = "../data/%s_%s_scans_proc" % (rad, date.strftime("%Y-%m-%d")) 
    pickle.dump(data, open(filename+".pickle", 'wb'))
    utils.to_remote_FS(conn, filename+".pickle", LFS, is_local_remove)
    return

""" Get data """

def to_pickle_files(dates = [dt.datetime(2010, 1, 15)], rads = ['bks'], conn=None, LFS=None, is_local_remove=True):
    for rad in rads:
        for date in dates:
            print("date, rad: ", date, rad)
            try:
                to_db(date, rad, conn, LFS, is_local_remove)
            except Exception: 
                print("Issue with data convertion:", rad, date)
                traceback.print_exc()
    return 0