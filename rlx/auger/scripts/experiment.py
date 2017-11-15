
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import rlxtools.auger as auger
import deepdish as dd
from rlxtools.ml import *
from sklearn.linear_model import LinearRegression                                                                                   
from sklearn.ensemble import RandomForestRegressor                     

import sys                                                                                              

for i in sys.argv[1:]:                                                     
    exec(i, globals())  

print

if not 'estimator' in locals():
   estimator = LinearRegression()
   print "using default estimator",
  
if not 'feature_set' in locals():
   feature_set = ['use_ref_rawgpserr']
   print "using default feature set"

if not 'n_ref_sds' in locals():
   n_ref_sds = 2
   print "using default n_ref_sds"

print
print "estimator is: ",  estimator                           
print "featureset is:", feature_set 
print "n_ref_sds is: ", n_ref_sds
print

print "decompressing data"
get_ipython().system(u'gunzip -c data/gps_prepared.hd5.gz > /tmp/gps-prepared.hd5')


print "reading data"
k = dd.io.load("/tmp/gps-prepared.hd5")
get_ipython().system(u'rm /tmp/gps-prepared.hd5')
edata = k["edata"]
vfree = k["vfree"]
vfixed = k["vfixed"]
xdfree = k["xdfree"]


# In[15]:


print "starting..."


# In[16]:


rcs = auger.explore_sds_combinations(estimator=estimator, edata=edata, n_jobs=1, n_ref_sds=n_ref_sds,
                                     feature_set=feature_set, test_period="2d", train_period="5d")

