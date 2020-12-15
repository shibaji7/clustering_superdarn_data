import datetime as dt
a_name="dbscan"
rads = ["bks"]
dates = [dt.datetime(2010, 1, 15)]

parameters = ['gate', 'beam', 'vel', 'wid', 'time', 'trad_gsflg', 'pow', 'clust_flg']
isgs={"case":0, "kind":0, "thresh":[0.5,0.5], "pth":0.5}
plot_params=["vel", "wid", "pow", "cluster", "isgs", "cum_isgs"]
save_params=["slist", "bmnum", "v", "w_l", "time", "p_l", "labels", "gflg_0", "gflg_1"]
plot_beams=[7]
save = True

def create_pickle_files():
    # Code to convert any day / radar to ".pickle" file for processing
    from pickle_creator import to_pickle_files
    import datetime as dt
    to_pickle_files(dates, rads)
    return

def run_gmm_algorithm():
    from statistics import run_algorithm
    import datetime as dt
    for date in dates:
        for rad in rads:
            run_algorithm(rad, date, date+dt.timedelta(days=1), a_name, gmm=False, 
              parameters = parameters, isgs=isgs, plot_beams=plot_beams, 
              plot_params=plot_params, save=save, save_params=save_params)
    return

if __name__ == "__main__":
    method = 2
    if method == 1: create_pickle_files()
    if method == 2: run_gmm_algorithm()
    pass