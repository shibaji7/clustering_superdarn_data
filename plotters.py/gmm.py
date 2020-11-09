#!/usr/bin/env python
# coding: utf-8

# # Gaussian Mixture Model
# 
# GMM runs on 5 features by default: beam, gate, time, velocity, and spectral  
# width. It performs well overall, even on clusters that are not well-separated   
# in space and time. However, it will often create clusters that are too high  
# variance,causing it to pull in scattered points that do not look like they  
# should be clustered together - see the fanplots in cluster.ipynb. It is also   
# slow, taking 5-10 minutes for one day of data.  
# 
# Parameters for each algorithm are set to default values (shown below), but can  
# be modified using the class constructor.
# 
# ### Optional arguments for GMM class constructor
# 
#      n_clusters=30
#          The number of GMM clusters to create.
#          
#      cov='full'
#          The covariance matrix to use for GMM.
#          See this post for more details: 
#          https://stats.stackexchange.com/questions/326671/different-covariance-types-for-gaussian-mixture-models
#          
#      features=['beam', 'gate', 'time', 'vel', 'wid'] 
#          Names of the features for GMM to run on. Can also include 'elv'.
#          
#      BoxCox=False
#          If BoxCox=True, 'wid' and 'vel' will be BoxCox transformed to   
#          convert them from an exponential distribution to a Gaussian.

# In[1]:

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.insert(0, '..')
from algorithms.gmm import GMM
import datetime
import numpy as np
import itertools

start_time = datetime.datetime(2017, 4, 4)
end_time = datetime.datetime(2017, 4, 5)
gmm = GMM(start_time, end_time, 'cvw', cov='full', n_clusters=10, BoxCox=True, load_model=False, save_model=False)
print(gmm.runtime)


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
# Make RTI plots to compare AJ's threshold with traditional threshold
#gmm.plot_rti(14, 'Ribiero')           # Slooow
# Make fanplots of the individual clusters over some time period
#fanplot_start = datetime.datetime(2017, 4, 4, 4, 0, 0)
#fanplot_end = datetime.datetime(2017, 4, 4, 4, 0, 0)
#gmm.plot_fanplots(fanplot_start, fanplot_end)


# In[ ]:

#lowest_bic = np.infty
#bic = []
#n_components_range = range(1, 31)
#cv_types = ['spherical', 'full']
#for cv_type in cv_types:
#    for n_components in n_components_range:
#        # Fit a Gaussian mixture with EM
#        gmm = GMM(start_time, end_time, 'cvw', cov=cv_type, n_clusters=n_components, BoxCox=True, load_model=False, save_model=False)
#        bic.append(gmm._bic)
#        if bic[-1] < lowest_bic:
#            lowest_bic = bic[-1]
#            best_gmm = gmm

#bic = np.array(bic)
#color_iter = itertools.cycle(['navy','darkorange'])
#clf = best_gmm
#bars = []

# Plot the BIC scores
#plt.figure(figsize=(5, 3))
#spl = plt.subplot(1, 1, 1)
#for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#    xpos = np.array(n_components_range) + .2 * (i - 2)
#    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#        (i + 1) * len(n_components_range)],
#        width=.2, color=color))
#plt.xticks(n_components_range[::3])
#plt.yscale("log")
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
#plt.title('BIC score per model')
#xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#        .2 * np.floor(bic.argmin() / len(n_components_range))
#plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
#spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)
#plt.savefig("../plots/bic.png", bbox_inches="tight")
