import pickle
from matplotlib.dates import date2num, num2date
import datetime as dt
import pandas as pd
import pdb
from get_sd_data import *


def to_db(date, rad):
    fdata = FetchData( rad, [date,
                date + dt.timedelta(days=1)] )
    _, scans = fdata.fetch_data(by="scan", scan_prop={"dur": 2, "stype": "themis"})
    #recs = fdata.convert_to_pandas(_beams_)

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
    
    bmax, nrang = 0, 0
    for i in range(len(scans)):
        s = scans[i]
        g, bm, v, w, el, tr, tm, n, ns, f, pw = [], [], [], [], [], [], [], [], [], [], []
        for b in s.beams:
            g.extend(np.array(b.slist).tolist())
            bm.extend([b.bmnum]*len(b.slist))
            v.extend(np.array(b.v).tolist())
            w.extend(np.array(b.w_l).tolist())
            tr.extend(np.array(b.gflg).tolist())
            el.extend(np.array(b.elv).tolist())
            tm.extend([date2num(b.time)]*len(b.slist))
            pw.extend(np.array(b.p_l).tolist())
        if i==0: 
            bmax = np.max(bm)
            nrang = b.nrang
        gate_scans.append(np.array(g))
        beam_scans.append(np.array(bm))
        vel_scans.append(np.array(v))
        wid_scans.append(np.array(w))
        time_scans.append(np.array(tm))
        trad_gs_flg_scans.append(np.array(tr))
        elv_scans.append(np.array(el))
        power_scans.append(np.array(pw))
    data = {'gate' : gate_scans, 'beam' : beam_scans, 'vel' : vel_scans, 'wid': wid_scans,
            'time' : time_scans, 'trad_gsflg' : trad_gs_flg_scans, 'elv': elv_scans, "pow":power_scans,
            'nrang' : nrang, 'nbeam' : bmax + 1}
    filename = "../data/%s_%s_scans" % (rad, date.strftime("%Y-%m-%d")) 
    pickle.dump(data, open(filename+".pickle", 'wb'))
    return

""" Get data """

def to_pickle_files(dates = [dt.datetime(2010, 1, 15)], rads = ['bks']):
    for rad in rads:
        for date in dates:
            print("date, rad: ", date, rad)
            to_db(date, rad)
    return