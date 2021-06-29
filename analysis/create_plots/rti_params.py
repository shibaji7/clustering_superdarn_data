import os
import sys
sys.path.append("./")
sys.path.append("create_plots/")
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.dates import num2date

import pydarn
from pysolar.solar import get_altitude
from astral import Observer, sun

from plotlib import RangeTimePlot as RTI
import utils
import rad_fov

from timezonefinder import TimezoneFinder

LFS = "LFS/LFS_clustering_superdarn_data/"

tf = TimezoneFinder(in_memory=True)

def get_sza(row, lats, lons):
    print(num2date(row.time), row.bmnum)
    lat, lon = lats[int(row["bmnum"]), int(row["slist"])],\
                    lons[int(row["bmnum"]), int(row["slist"])]
    dn = num2date(row["time"])
    d = dn.replace(tzinfo=dt.timezone.utc)
    sza = 90.-get_altitude(lat, lon, d)
    return sza


def get_sunrise_sunset(lat, lon, d):
    d = d.replace(tzinfo=dt.timezone.utc)
    local_time_zone = tf.timezone_at(lng=lon, lat=lat)
    #cal, togg = 0, True
    #for mins in range(1440):
    #    dn = d + dt.timedelta(minutes=mins)
    #    sza = 90.-get_altitude(lat, lon, dn)
    #    if int(sza/110.) == 1 and togg: sunset = dn; togg = False
    #    elif int(sza/90.) == 0 and not togg: sunrise = dn; break
    #print(sunrise, sunset)
    o = Observer(lat, lon)
    s = sun.sun(o, d, tzinfo=local_time_zone)
    sunrise, sunset = s["sunrise"], s["sunset"]
    print(sunrise, sunset)
    return sunrise, sunset


gmm = True
a_name = "dbscan"
rads = ["cvw"]
dates = [dt.datetime(2012,1,2)]
beam = 11
remove_file = False
maxGate = None

pubfile = utils.get_pubfile()
conn = utils.get_session(key_filename=pubfile)

if gmm: fname = "../outputs/figures_for_papers/{rad}.{a_name}.gmm.{dn}.csv"
else: fname = "../outputs/figures_for_papers/{rad}.{a_name}.{dn}.csv"
X = pd.DataFrame()

for rad, dn in zip(rads, dates):
    floc = fname.format(rad=rad, a_name=a_name, dn=dn.strftime("%Y%m%d"))
    if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
    X = pd.read_csv(floc)
    hdw = pydarn.read_hdw_file(rad)
    egate = hdw.gates if not maxGate else maxGate
    rfov = rad_fov.CalcFov(hdw=hdw, ngates=egate)
    if "sza" not in X.columns: 
        X["sza"] = X.apply(lambda r: get_sza(r, rfov.latFull, rfov.lonFull), axis=1)
        X.to_csv(floc)

beam_lats, beam_lons = rfov.latFull[beam, :], rfov.lonFull[beam, :]
sunrise, sunset = [], []
for lat, lon in zip(beam_lats, beam_lons):
    sr, ss = get_sunrise_sunset(lat, lon, dn)
#    sunrise.append(sr)
#    sunset.append(ss)
sun_rise_set_obj = {"gate": range(len(beam_lons)), "sunset": [sunset]*len(beam_lons), "sunrise": [sunrise]*len(beam_lons)}
conn.close()
if remove_file: os.system("rm -rf ../outputs/cluster_tags/*")
rti = RTI(100, np.unique(X.time), "", num_subplots=3)
rti.addParamPlot(X, beam, "Date: %s, %s, Beam: %d"%(dates[0].strftime("%Y-%m-%d"), rad.upper(), beam), p_max=50, p_min=-50, 
                 p_step=20, xlabel="", zparam="v", label="Velocity [m/s]", ss_obj=None)
rti.addParamPlot(X, beam, "", p_max=33, p_min=3, p_step=6, xlabel="", zparam="p_l", label="Power [db]", ss_obj=None)
rti.addParamPlot(X, beam, "", p_max=50, p_min=0, p_step=10, xlabel="Time [UT]", zparam="w_l", 
                 label="Spect. Width [m/s]", ss_obj=None)
rti.save("create_plots/images/params.png")