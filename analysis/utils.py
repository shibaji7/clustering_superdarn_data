#/usr/bin/env python

"""
utility.py: module is dedicated to skills of a clustering algoritm and other utilities.

    Internal Validation class (Skill)
"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import cohen_kappa_score
from scipy.stats import beta

import paramiko
import os
from cryptography.fernet import Fernet
import json

class Skills(object):
    """
    Internal Validation class that computes model skills.
    
    Internal validation methods make it possible to establish
    the quality of the clustering structure without having access
    to external information (i.e. they are based on the information
    provided by data used as input to the clustering algorithm).
    """
    
    def __init__(self, X, labels, verbose=True):
        """
        Initialize the parameters.
        X: List of n_features-dimensional data points. Each row corresponds to a single data point.
        lables: Predicted labels for each sample.
        """
        self.X = X
        self.labels = labels
        self.clusters = set(self.labels)
        self.n_clusters = len(self.clusters)
        self.n_samples, _ = X.shape
        self.verbose = verbose
        self._compute()
        return

    def _compute(self):
        """
        Compute all different skills scores
        - Calinski Harabasz Score (chscore)
        - Ball Hall Score (bhscore)
        - Hartigan Score (hscore)
        - Xu Score (xuscore)
        """
        self._errors_mse()
        self._errors_mdse()
        self.chscore = calinski_harabasz_score(self.X, self.labels)
        self.bhscore, self.mdbhscore = self._ball_hall_score()
        self.hscore, self.mdhscore = self._hartigan_score()
        self.xuscore, self.mdxuscore = self._xu_score()
        if self.verbose:
            print("\n Estimated Skills (MSE).")
            print(" Calinski Harabasz Score - ",self.chscore)
            print(" Ball-Hall Score - ",self.bhscore)
            print(" Hartigan Score - ",self.hscore)
            print(" Xu Score - ",self.xuscore)
            print("\n Estimated Skills (MdSE).")
            print(" Ball-Hall Score - ",self.mdbhscore)
            print(" Hartigan Score - ",self.mdhscore)
            print(" Xu Score - ",self.mdxuscore)
            print(" Estimation done.")
        return

    def _errors_mse(self):
        """
        Estimate SSE and SSB of the model.
        """
        sse, ssb = 0., 0.
        mean = np.mean(self.X, axis=0)
        for k in self.clusters:
            _x = self.X[self.labels == k]
            mean_k = np.mean(_x, axis=0)
            ssb += len(_x) * np.sum((mean_k - mean) ** 2)
            sse += np.sum((_x - mean_k) ** 2)
        self.sse, self.ssb = sse, ssb
        return
    
    def _errors_mdse(self):
        """
        Estimate Median SSE and SSB of the model.
        """
        mdsse, mdssb = 0., 0.
        med = np.median(self.X, axis=0)
        for k in self.clusters:
            _x = self.X[self.labels == k]
            med_k = np.median(_x, axis=0)
            mdssb += len(_x) * np.sum((med_k - med) ** 2)
            mdsse += np.sum((_x - med_k) ** 2)
        self.mdsse, self.mdssb = mdsse, mdssb
        return

    def _ball_hall_score(self):
        """
        The Ball-Hall index is a dispersion measure based on the quadratic
        distances of the cluster points with respect to their centroid.
        """
        n_clusters = len(set(self.labels))
        return self.sse / n_clusters, self.mdsse / n_clusters
    
    def _hartigan_score(self):
        """
        The Hartigan index is based on the logarithmic relationship between
        the sum of squares within the cluster and the sum of squares between clusters.
        """
        return np.log(self.ssb/self.sse), np.log(self.mdssb/self.mdsse)

    def _xu_score(self):
        """
        The Xu coefficient takes into account the dimensionality D of the data,
        the number N of data examples, and the sum of squared errors SSEM form M clusters.
        """
        n_clusters = len(set(self.labels))
        xuscore = np.log(n_clusters) + self.X.shape[1] * np.log2(np.sqrt(self.sse/(self.X.shape[1]*self.X.shape[0]**2)))
        mdxuscore = np.log(n_clusters) + self.X.shape[1] * np.log2(np.sqrt(self.mdsse/(self.X.shape[1]*self.X.shape[0]**2)))
        return xuscore, mdxuscore

def get_kappa(y1, y2):
    k = cohen_kappa_score(y1, y2)
    print("Kappa:", cohen_kappa_score(y1, y2))
    return k

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v"):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    plotParamDF[xparam] = plotParamDF[xparam].tolist()
    plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

class ScatterTypeDetection(object):
    """ Detecting scatter type """

    def __init__(self, df):
        """ kind: 0- individual, 2- KDE by grouping """
        self.df = df.fillna(0)
        return

    def run(self, cases=[0,1,2], thresh=[1./3.,2./3.], pth=0.5):
        self.pth = pth
        self.thresh = thresh
        self.clusters = {}
        for cs in cases:
            self.case = cs
            self.clusters[self.case] = {}
            for kind in range(2):
                if kind == 0: 
                    self.indp()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                if kind == 1: 
                    self.kde()
                    self.df["gflg_%d_%d"%(cs,kind)] = self.gs_flg
                    self.df["proba_%d"%(cs)] = self.proba
        return self.df.copy(), self.clusters
    
    def ribiero_gs_flg(self, vel, time):
        L = np.abs(time[-1] - time[0]) * 24
        high = np.sum(np.abs(vel) > 15.0)
        low = np.sum(np.abs(vel) <= 15.0)
        if low == 0: R = 1.0  # TODO hmm... this works right?
        else: R = high / low  # High vel / low vel ratio
        # See Figure 4 in Ribiero 2011
        if L > 14.0:
            # Addition by us
            if R > 0.15: return False    # IS
            else: return True     # GS
            # Classic Ribiero 2011
            #return True  # GS
        elif L > 3:
            if R > 0.2: return False
            else: return True
        elif L > 2:
            if R > 0.33: return False
            else: return True
        elif L > 1:
            if R > 0.475: return False
            else: return True
        # Addition by Burrell 2018 "Solar influences..."
        else:
            if R > 0.5: return False
            else: return True
        # Classic Ribiero 2011
        # else:
        #    return False

    def kde(self):
        from scipy.stats import beta
        import warnings
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["labels"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        self.proba = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            self.clusters[self.case][bm] = {}
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams) 
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    if self.case == 0: f = 1/(1+np.exp(np.abs(v)+w/3-30))
                    if self.case == 1: f = 1/(1+np.exp(np.abs(v)+w/4-60))
                    if self.case == 2: f = 1/(1+np.exp(np.abs(v)-0.139*w+0.00113*w**2-33.1))
                    ## KDE part
                    f[f==0.] = np.random.uniform(0.001, 0.01, size=len(f[f==0.]))
                    f[f==1.] = np.random.uniform(0.95, 0.99, size=len(f[f==1.]))
                    try:
                        a, b, loc, scale = beta.fit(f, floc=0., fscale=1.)
                        auc = 1 - beta.cdf(self.pth, a, b, loc=0., scale=1.)
                        if np.isnan(auc): auc = np.mean(f)
                        #print(" KDE Estimating AUC: %.2f"%auc, c)
                    except:
                        auc = np.mean(f)
                        #print(" Estimating AUC: %.2f"%auc, c)
                    if auc < self.thresh[0]: gflg, ty = 0., "IS"
                    elif auc >= self.thresh[1]: gflg, ty = 1., "GS"
                    else: gflg, ty = -1, "US"
                    self.gs_flg[clust_mask] = gflg
                    self.proba[clust_mask] = f
                    self.clusters[self.case][bm][c] = {"auc": auc, "type": ty}
        return

    def indp(self):
        vel = np.hstack(self.df["v"])
        wid = np.hstack(self.df["w_l"])
        clust_flg_1d = np.hstack(self.df["labels"])
        self.gs_flg = np.zeros(len(clust_flg_1d))
        beams = np.hstack(self.df["bmnum"])
        for bm in np.unique(beams):
            for c in np.unique(clust_flg_1d):
                clust_mask = np.logical_and(c == clust_flg_1d, bm == beams)
                if c == -1: self.gs_flg[clust_mask] = -1
                else:
                    v, w = vel[clust_mask], wid[clust_mask]
                    gflg = np.zeros(len(v))
                    if self.case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
                    if self.case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
                    if self.case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
                    self.gs_flg[clust_mask] = gflg
        return
    
    @staticmethod
    def indp_classical(w, v, case=2):
        if case == 0: gflg = (np.abs(v)+w/3 < 30).astype(int)
        if case == 1: gflg = (np.abs(v)+w/4 < 60).astype(int)
        if case == 2: gflg = (np.abs(v)-0.139*w+0.00113*w**2<33.1).astype(int)
        return gflg
    
    
class Conn2Remote(object):
    
    def __init__(self, host, user, key_filename, port=22, passcode=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.passcode = passcode
        self.port = port
        self.con = False
        if passcode: self.decrypt()
        self.conn()
        return
    
    def decrypt(self):
        passcode = bytes(self.passcode, encoding="utf8")
        cipher_suite = Fernet(passcode)
        self.user = cipher_suite.decrypt(bytes(self.user, encoding="utf8")).decode("utf-8")
        self.host = cipher_suite.decrypt(bytes(self.host, encoding="utf8")).decode("utf-8")
        return
    
    def conn(self):
        if not self.con:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname=self.host, port = self.port, username=self.user, key_filename=self.key_filename)
            self.scp = paramiko.SFTPClient.from_transport(self.ssh.get_transport())
            self.con = True
        return
    
    def close(self):
        if self.con:
            self.scp.close()
            self.ssh.close()
        return
    
def encrypt(host, user, filename="config/passcode.json"):
    passcode = Fernet.generate_key()
    cipher_suite = Fernet(passcode)
    host = cipher_suite.encrypt(bytes(host, encoding="utf8"))
    user = cipher_suite.encrypt(bytes(user, encoding="utf8"))
    with open(filename, "w") as f:
        f.write(json.dumps({"user": user.decode("utf-8"), "host": host.decode("utf-8"), "passcode": passcode.decode("utf-8")},
                           sort_keys=True, indent=4))
    return

def get_session(filename="config/passcode.json", key_filename="", isclose=False):
    with open(filename, "r") as f:
        obj = json.loads("".join(f.readlines()))
        conn = Conn2Remote(obj["host"], obj["user"], 
                           key_filename=key_filename, 
                           passcode=obj["passcode"])
    if isclose: conn.close()    
    return conn

def chek_remote_file_exists(fname, conn):
    try:
        print("File Check:",fname)
        conn.scp.stat(fname)
        return True
    except FileNotFoundError:
        return False

def to_remote_FS(conn, local_file, LFS, is_local_remove=False):
    remote_file = LFS + local_file.replace("../", "")
    print(" To file:", remote_file)
    conn.scp.put(local_file, remote_file)
    if is_local_remove: os.remove(local_file)
    return

def from_remote_FS(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    print(" From file:", remote_file)
    conn.scp.get(remote_file, local_file)
    return

def to_remote_FS_dir(conn, ldir, local_file, LFS, is_local_remove=False):
    remote_file = LFS + local_file.replace("../", "")
    rdir = LFS + ldir.replace("../", "")
    conn.ssh.exec_command("mkdir -p " + rdir)
    print(" To file:", remote_file)
    conn.scp.put(local_file, remote_file)
    if is_local_remove: os.remove(local_file)
    return

def get_pubfile():
    with open("config/pub.json", "r") as f:
        obj = json.loads("".join(f.readlines()))
        pubfile = obj["pubfile"]
    return pubfile

def fetch_file(conn, local_file, LFS):
    remote_file = LFS + local_file.replace("../", "")
    is_remote = chek_remote_file_exists(remote_file, conn)
    if is_remote: from_remote_FS(conn, local_file, LFS)
    return is_remote
    
def ribiero_gs_flg(vel, time):
    L = np.abs(time[-1] - time[0]) * 24
    high = np.sum(np.abs(vel) > 15.0)
    low = np.sum(np.abs(vel) <= 15.0)
    if low == 0: R = 1.0  # TODO hmm... this works right?
    else: R = high / low  # High vel / low vel ratio
    # See Figure 4 in Ribiero 2011
    if L > 14.0:
        # Addition by us
        if R > 0.15: return False    # IS
        else: return True     # GS
        # Classic Ribiero 2011
        #return True  # GS
    elif L > 3:
        if R > 0.2: return False
        else: return True
    elif L > 2:
        if R > 0.33: return False
        else: return True
    elif L > 1:
        if R > 0.475: return False
        else: return True
    # Addition by Burrell 2018 "Solar influences..."
    else:
        if R > 0.5: return False
        else: return True
    # Classic Ribiero 2011
    # else:
    #    return False

def _run_riberio_threshold(u, beam):
    df = u[u.bmnum==beam]
    clust_flag = np.array(df.labels); gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v)); t = np.hstack(df.time)
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1: gs_flg[clust_mask] = -1
        else: gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
    df["ribiero_gflg"] = gs_flg
    return df

def _run_riberio_threshold_on_rad(u):
    df = u.copy()
    clust_flag = np.array(df.labels); gs_flg = np.zeros_like(clust_flag)
    vel = np.hstack(np.abs(df.v)); t = np.hstack(df.time)
    gs_flg = np.zeros_like(clust_flag)
    for c in np.unique(clust_flag):
        clust_mask = c == clust_flag
        if c == -1: gs_flg[clust_mask] = -1
        else: gs_flg[clust_mask] = ribiero_gs_flg(vel[clust_mask], t[clust_mask])
    df["ribiero_gflg"] = gs_flg
    return df

if __name__ == "__main__":
    encrypt("cascades1.arc.vt.edu", "shibaji7")
    #get_session(isclose=True)
    pass