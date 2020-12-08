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
        self._errors()
        self.chscore = calinski_harabasz_score(self.X, self.labels)
        self.bhscore = self._ball_hall_score()
        self.hscore = self._hartigan_score()
        self.xuscore = self._xu_score()
        if self.verbose:
            print("\n Estimated Skills.")
            print(" Calinski Harabasz Score - ",self.chscore)
            print(" Ball-Hall Score - ",self.bhscore)
            print(" Hartigan Score - ",self.hscore)
            print(" Xu Score - ",self.xuscore)
            print(" Estimation done.")
        return

    def _errors(self):
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

    def _ball_hall_score(self):
        """
        The Ball-Hall index is a dispersion measure based on the quadratic
        distances of the cluster points with respect to their centroid.
        """
        n_clusters = len(set(self.labels))
        return self.sse / n_clusters
    
    def _hartigan_score(self):
        """
        The Hartigan index is based on the logarithmic relationship between
        the sum of squares within the cluster and the sum of squares between clusters.
        """
        return np.log(self.ssb/self.sse)

    def _xu_score(self):
        """
        The Xu coefficient takes into account the dimensionality D of the data,
        the number N of data examples, and the sum of squared errors SSEM form M clusters.
        """
        n_clusters = len(set(self.labels))
        return np.log(n_clusters) + self.X.shape[1] * np.log2(np.sqrt(self.sse/(self.X.shape[1]*self.X.shape[0]**2)))

    def get_kappa(self, y1, y2):
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

    def run(self, case=0, thresh=[1./3.,2./3.], pth=0.5):
        self.pth = pth
        self.thresh = thresh
        self.case = case
        self.clusters = {}
        for kind in range(2):
            if kind == 0: 
                self.indp()
                self.df["gflg_%d"%kind] = self.gs_flg
            if kind == 1: 
                self.kde()
                self.df["gflg_%d"%kind] = self.gs_flg
                self.df["proba"] = self.proba
        return self.df.copy(), self.clusters

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
                    self.clusters[c] = {"auc": auc, "type": ty}
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