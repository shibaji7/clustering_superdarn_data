# Import python packages
import ipdb
import numpy as np
import pandas as pd
import datetime as dt
import sqlite3 as sql

# Import DaViTpy packages
import davitpy.pydarn.sdio as sdio
import davitpy.pydarn.proc.fov.update_backscatter as ub

#--------------------------------------------------------------------------
# Define the colors (can be overwritten)
clinear = "Spectral_r"
ccenter = "Spectral"

morder =  {"region":{"D":0, "E":1, "F":2},
           "reg":{"0.5D":0, "1.0D":1, "0.5E":2, "1.0E":3, "1.5E":4, "2.0E":5,
                  "2.5E":6, "3.0E":7, "0.5F":8, "1.0F":9, "1.5F":10, "2.0F":11,
                  "2.5F":12, "3.0F":13},
           "hop":{0.5:0, 1.0:1, 1.5:2, 2.0:3, 2.5:4, 3.0:5},}
mc = {"region":{"D":"g", "E":"m", "F":"b"},
      "reg":{"0.5D":"g", "1.0D":"y", "0.5E":"r", "1.0E":"m",
             "1.5E":(1.0, 0.5, 1.0), "2.0E":(0.5, 0, 0.25),
             "2.5E":(1.0,0.7,0.2), "3.0E":(0.5, 0, 0.1),
             "0.5F":(0.0, 0.0, 0.75), "1.0F":(0.0, 0.5, 0.5), "1.5F":"b",
             "2.0F":"c", "2.5F":(0.25, 0.75, 1.0), "3.0F":(0.0, 1.0, 1.0)},
      "hop":{0.5:"b", 1.0:"r", 1.5:"c", 2.0:"m",
             2.5:(0.25, 0.75, 1.0), 3.0:(0.5, 0, 0.25)},}
mm = {"region":{"D":"d", "E":"o", "F":"^", "all":"|"},
      "reg":{"0.5D":"d", "1.0D":"Y", "0.5E":"^", "1.0E":"o", "1.5E":">",
             "2.0E":"8", "2.5E":"*", "3.0E":"H", "0.5F":"v", "1.0F":"p",
             "1.5F":"<", "2.0F":"h", "2.5F":"D", "3.0F":"s", "all":"|"},
      "hop":{0.5:"d", 1.0:"o", 1.5:"s", 2.0:"^", 2.5:"v", 3.0:"p",
             "all":"|"},}


def update_gsflg(stime=dt.datetime(2017,9,20,0),etime=dt.datetime(2017,9,20,1),
		rad="sas", file_type="fitacf",
		min_pnts=3, region_hmin={"D":75.0,"E":115.0,"F":150.0},
		region_hmax={"D":115.0,"E":150.0,"F":900.0},
		rg_box=[2,5,10,20,25], vh_box=[50.0,50.0,50.0,150.0,150.0],
		max_rg=[5,25,40,76,226], max_hop=3.0,
		ut_box=dt.timedelta(minutes=20.0),tdiff=None,
		tdiff_args=list(), tdiff_e=None, tdiff_e_args=list(),
		ptest=True, step=6, strict_gs=False):
    """Routines to update the groundscatter and elevation angle, as well as 
	determine the virtual height, hop, and origin field-of-view for each backscatter point.

    Parameters
    -----------
    stime : (dt.datetime)
        Starting time of plot (will pad loaded data).
    etime : (dt.datetime)
        Ending time of plot (will pad loaded data).
    rad : (str)
        Radar code name.
    file_type : (str)
        Type of data file to download (default="fitacf")
    min_pnts : (int)
        The minimum number of points necessary to perform certain range gate
        or beam specific evaluations. (default=3)
    region_hmax : (dict)
        Maximum virtual heights allowed in each ionospheric layer.
        (default={"D":115.0,"E":150.0,"F":900.0})
    region_hmin : (dict)
        Minimum virtual heights allowed in each ionospheric layer.
        (default={"D":75.0,"E":115.0,"F":150.0})
    rg_box : (list of int)
        The total number of range gates to include when examining the elevation
        angle across all beams. (default=[2,5,10,20])
    vh_box : (list of float)
        The total width of the altitude box to consider when examining the
        elevation angle across all beams at a given range gate.
        (default=[50.0,50.0,50.0,150.0])
    max_rg : (list)
        Maximum range gate to use each range gate and virtual height at
        (default=[5,25,40,76])
    max_hop : (float)
        Maximum hop that the corresponding rg_box and vh_box values applies
        to.  (default=3.0)
    ut_box : (class dt.timedelta)
        Total width of universal time box to examine for backscatter FoV
        continuity. (default=20.0 minutes)
    tdiff : (function or NoneType)
        A function to retrieve tdiff values (in microsec) using the radar ID
        number current datetime, and transmisson frequency as input.
        Additional inputs may be specified using tdiff_args.  Example:
        def get_tdiff(stid, time, tfreq, filename) { do things } return tdiff
        tdiff=get_tdiff, tdiff_args=["tdiff_file"]
        (default=None)
    tdiff_args : (list)
        A list specifying any arguements other than radar, time, and
        transmission frequency to run the specified tdiff function.
        (default=list())
    tdiff_e : function or NoneType)
        A function to retrieve tdiff error values (in microsec) using the radar
        ID number, current datetime, and transmisson frequency as input.
        Additional inputs may be specified using tdiff_e_args.  Example:
        def get_tdiffe(stud, time, tfreq, filename) { do things } return tdiffe
        tdiff_e=get_tdiffe, tdiff_e_args=["tdiff_file"]
        (default=None)
    tdiff_e_args : (list)
        A list specifying any arguements other than radar, time, and
        transmission frequency to run the specified tdiff_e function.
        (default=list())
    ptest : (boolian)
        Test to see if a propagation path is realistic (default=True)
    step : (int)
        Level of processing to perform (1-6).  6 performs all steps. (default=6)
    strict_gs : (boolian)
        Remove indeterminately flagged backscatter (default=True)

    Returns
    ---------
    beams : (list)
        A dictionary of updated beamData class objects.  The dictionary keys
        correspond to the beam numbers, and contain np.arrays of beams sorted
        by UT with the following additional/updated attributes

        beam.fit.fovelv : added : Accounts for adjusted tdiff and origin FoV
        beam.fit.fovelv_e : added : elevation error
        beam.fit.felv : added : Elevation angle assuming front FoV
        beam.fit.felv_e : added : Elevation angle error assuming front FoV
        beam.fit.belv : added : Elevation angle assuming rear FoV
        beam.fit.belv_e : added : Elevation angle error assuming front FoV
        beam.fit.vheight : added : virtual height of ionosphere in km
        beam.fit.vheight_e : added : error in virtual height (km)
        beam.fit.fvheight : added : virtual height assuming front FoV
        beam.fit.fvheight_e : added : error in virtual height assuming front FoV
        beam.fit.bvheight : added : virtual height assuming rear FoV
        beam.fit.bvheight_e : added : error in virtual height assuming rear FoV
        beam.fit.hop : added : Hop assuming the assigned origin FoV
        beam.fit.fhop : added : Hop assuming the front FoV
        beam.fit.bhop : added : Hop assuming the rear FoV
        beam.fit.region : added : Region assuming the assigned origin FoV
        beam.fit.fregion : added : Region assuming the front FoV
        beam.fit.bregion : added : Region assuming the rear FoV
        beam.fit.fovflg : added : Flag indicating origin FoV (1=front, -1=back,
                                  0=indeterminate)
        beam.fit.pastfov : added : Flag indicating past FoV assignments
        beam.fit.gflg : updated : Flag indicating backscatter type
                                  (1=ground, 0=ionospheric, -1=indeterminate)
        beam.prm.tdiff : added : tdiff used in elevation (microsec)
        beam.prm.tdiff_e : possibly added : tdiff error (microsec)
    """


    beams = dict()
    # Load data for this radar, padding data based on the largest
    # temporal boxcar window used in the FoV processing
    rad_ptr = sdio.radDataRead.radDataOpen(stime-ut_box, rad,
                                           eTime=etime+ut_box,
                                           #cp=rad_cp[rad],
                                           fileType=file_type)

    # Process the beams for this radar
    beams = ub.update_backscatter(rad_ptr, min_pnts=min_pnts,
                                       region_hmax=region_hmax,
                                       region_hmin=region_hmin,
                                       rg_box=rg_box, vh_box=vh_box,
                                       max_rg=max_rg, max_hop=max_hop,
                                       ut_box=ut_box, tdiff=tdiff,
                                       tdiff_args=tdiff_args,
                                       tdiff_e=tdiff_e,
                                       tdiff_e_args=tdiff_e_args,
                                       ptest=ptest, strict_gs=strict_gs,
                                       step=step)

    return beams


def read_save_sd_db(rad, stm, etm):

	""" read the updated data of one radar from the Virginia Tech sever
		and save to a database 
	"""
	
	ds_file = "/home/xueling/data/sqlite3/"+rad+"_GSoC_"+stm.strftime("%Y-%m-%d")+".db"
	time, freq, nrang, beam, num_scatter, cpid =  [], [], [], [], [], []
	gate, elev, vel, wid, power =  [], [], [], [], []
	updated_gsflg, hop, region, vheight, vheight_e, elv_e = [], [], [], [], [], []
	phi0, rsep, frang, nave, nsky, nsch = [], [], [], [], [], []


	data_dict = dict()

	#update the beam data to include the emprical model results
	updated_beams = update_gsflg(stime=stm,etime=etm,rad=rad)

	for bm_num in updated_beams.keys():
		if len(updated_beams[bm_num]) > 0:
			#print len(updated_beams[bm_num])
			for i in range(len(updated_beams[bm_num])):
				if (updated_beams[bm_num][i] is not None and updated_beams[bm_num][i].time >= stm and updated_beams[bm_num][i].time <= etm):
					if updated_beams[bm_num][i].fit.slist is not None:
						power_filter = updated_beams[bm_num][i].fit.p_l
						elv_filter = updated_beams[bm_num][i].fit.fovelv
						#remove data with power <= 6 or fovelv == nan
						ind = np.array([j for j, (pw,elv) in enumerate(zip(power_filter,elv_filter)) if (pw > 6 and not np.isnan(elv))]) 

						#ipdb.set_trace()
						if ind.size:
							time.append(updated_beams[bm_num][i].time)
							freq.append(updated_beams[bm_num][i].prm.tfreq / 1e3)
							nrang.append(updated_beams[bm_num][i].prm.nrang)
							beam.append(updated_beams[bm_num][i].bmnum)
							num_scatter.append(len(ind))
							cpid.append(updated_beams[bm_num][i].cp)

							gate.append(np.array(updated_beams[bm_num][i].fit.slist)[ind].tolist())
							elev.append(np.array(updated_beams[bm_num][i].fit.fovelv)[ind].tolist())  #added
							vel.append(np.array(updated_beams[bm_num][i].fit.v)[ind].tolist())
							wid.append(np.array(updated_beams[bm_num][i].fit.w_l)[ind].tolist())
							power.append(np.array(updated_beams[bm_num][i].fit.p_l)[ind].tolist())

							updated_gsflg.append(np.array(updated_beams[bm_num][i].fit.gflg)[ind].tolist()) #updated
							hop.append(np.array(updated_beams[bm_num][i].fit.hop)[ind].tolist())
							region.append(np.array(updated_beams[bm_num][i].fit.region)[ind].tolist())
							vheight.append(np.array(updated_beams[bm_num][i].fit.vheight)[ind].tolist())
							vheight_e.append(np.array(updated_beams[bm_num][i].fit.vheight_e)[ind].tolist())
							elv_e.append(np.array(updated_beams[bm_num][i].fit.fovelv_e)[ind].tolist())


							phi0.append(np.array(updated_beams[bm_num][i].fit.phi0)[ind].tolist())
							rsep.append(updated_beams[bm_num][i].prm.rsep)
							frang.append(updated_beams[bm_num][i].prm.frang)
							nave.append(updated_beams[bm_num][i].prm.nave)
							nsky.append(updated_beams[bm_num][i].prm.noisesky)
							nsch.append(updated_beams[bm_num][i].prm.noisesearch)


	data_dict = {"time":time, "frequency":freq, "nrang":nrang, "beam":beam, "num_scatter":num_scatter, \
				"cpid":cpid, "gate":gate, "elevation":elev, "velocity":vel, "width":wid, "power":power, \
				"hop":hop, "region":region, "vheight":vheight, "vheight_e":vheight_e, "elv_e":elv_e, \
				"gsflg":updated_gsflg,"phi0":phi0, "rsep":rsep, "frang":frang , "nave":nave, "nsky":nsky, "nsch":nsch}


	#save to the df_file database
	df = pd.DataFrame(data_dict) 

	if len(df) > 0:   
		#print len(df)
		df.gate = [str(x) for x in df.gate]
		df.velocity = [str(x) for x in df.velocity]
		df.phi0 = [str(x) for x in df.phi0] 
		df.power = [str(x) for x in df.power]
		df.width = [str(x) for x in df.width] 
		df.gsflg = [str(x) for x in df.gsflg]
		df.elevation = [str(x) for x in df.elevation]
		df.hop = [str(x) for x in df.hop]
		df.region = [str(x) for x in df.region]
		df.vheight = [str(x) for x in df.vheight]
		df.vheight_e = [str(x) for x in df.vheight_e]
		df.elv_e = [str(x) for x in df.elv_e]

		#if os.path.exists(ds_file): os.remove(ds_file)
		print "Data file - ", ds_file
		conn = sql.connect(ds_file)
		df.to_sql("sd_table_"+rad,con=conn,if_exists="append",index=False)
		conn.commit()      
		conn.close()

	return


def save_sd_db_cvw():

	""" save to the database daily
	"""

	rad = "cvw"

	#sdate = dt.datetime(2015,5,1) #20150501 to 20150512
	#num_days = 12

	#sdate = dt.datetime(2015,6,1) #20150601 to 20150609
	#num_days = 9

	#sdate = dt.datetime(2015,7,1) #20150701 to 20150721
	#num_days = 21

	#sdate = dt.datetime(2015,10,1)  #20151001 to 20151016
	#num_days = 16

	#sdate = dt.datetime(2015,10,30) #20151030 to 20151112
	#num_days = 14

	#sdate = dt.datetime(2015,11,29) #20151129 to 20151204
	#num_days = 6

	#sdate = dt.datetime(2016,1,1) #20160101 to 20160106
	#num_days = 6

	#sdate = dt.datetime(2016,3,1) #20160301 to 20160307
	#num_days = 7

	#sdate = dt.datetime(2016,4,11) #20160411 to 20160423
	#num_days = 13

	#sdate = dt.datetime(2016,8,23) #20160823 to 20160906
	#num_days = 15

	#sdate = dt.datetime(2016,9,27) #20160927 to 20161003
	#num_days = 7

	#for d in range(num_days):
	#	stm = sdate+dt.timedelta(days=d)
	#	etm = stm+dt.timedelta(days=1)
	#	read_save_sd_db(rad, stm, etm)


	sdates = [dt.datetime(2015,5,1),dt.datetime(2015,6,1),dt.datetime(2015,7,1),dt.datetime(2015,10,1),
				dt.datetime(2015,10,30),dt.datetime(2015,11,29),dt.datetime(2016,1,1),dt.datetime(2016,3,1),
				dt.datetime(2016,4,11),dt.datetime(2016,8,23),dt.datetime(2016,9,27)]
	num_days = [12,9,21,16,14,6,6,7,13,15,7]

	for ind, sdate in enumerate(sdates):
		for d in range(num_days[ind]):
			stm = sdate+dt.timedelta(days=d)
			etm = stm+dt.timedelta(days=1)
			read_save_sd_db(rad, stm, etm)


def save_sd_db_daily_file():

	""" save to the database daily
	"""
	rad = "bks"
	#rad = "sas"

	#sdates = [dt.datetime(2017,1,17),dt.datetime(2017,3,13),dt.datetime(2017,4,4),dt.datetime(2017,5,30),
	#			dt.datetime(2017,8,20),dt.datetime(2017,9,20),dt.datetime(2017,10,16),dt.datetime(2017,11,14),
	#			dt.datetime(2017,12,8),dt.datetime(2017,12,17),dt.datetime(2017,12,18),dt.datetime(2017,12,19),
	#			dt.datetime(2018,1,25),dt.datetime(2018,2,7),dt.datetime(2018,2,8),dt.datetime(2018,3,8),dt.datetime(2018,4,5)]
        sdates = [dt.datetime(2015,3,17)]
	for sdate in sdates:
		stm = sdate
		etm = stm+dt.timedelta(days=1)
		read_save_sd_db(rad, stm, etm)

save_sd_db_daily_file()
