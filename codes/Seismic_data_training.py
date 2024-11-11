#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:14:52 2024

@author: sebinjohn
"""

import os
os.chdir("/Users/sebinjohn/ML_proj/codes/")
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from obspy import UTCDateTime
from medfilt import medfilt
import xarray as xr



st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1

time_frame=[]
for i in range (int((et-st)/(3600*windw))):
    time_frame.append(st)
    st=st+(3600*windw)
    
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/Video Making")
freq=[]
name= pd.read_xml("pdf0.xml", xpath="/PsdRoot/Psds[1]/Psd[1]/value[@freq]")
for i in range (96):
    freq.append(name.iloc[i]['freq'])

freq=np.array(freq)

def mean(res):
    mea=np.mean(res,axis=0)
    return mea


stations=[]
inde=-1
final_seis_ML=np.zeros((15,len(time_frame)))
for files in glob.glob("/Users/sebinjohn/AON_PROJECT/Data/ML_seismic_train/*/*.npy"):
    print(files)
    stations.append(files.split("/")[-2])
    inde+=1
    seis=np.load(files)
    res=seis[53:61,:]
    out_mean=mean(res)
    out_mean_med=medfilt(out_mean,7)
    interp_inde=np.array([])
    interp_x=out_mean_med.copy()
    datagap=np.where(interp_x==0)[0]
    data_x=np.where(interp_x!=0)[0]
    for i in range(len(data_x)-1):
        if data_x[i+1]-data_x[i]<12 and data_x[i+1]-data_x[i]>1:
            interp_inde=np.append(interp_inde,np.array( [i for i in range(int(data_x[i])+1,int(data_x[i+1]))]))
        else:
            continue
    if len(interp_inde)>1:
        interp=np.interp(interp_inde, data_x.reshape(np.shape(data_x)[0]), interp_x[data_x].reshape(np.shape(data_x)[0]))
        interp_inde=(interp_inde+1).astype("int32")
        interp_x[interp_inde]=interp
    else:
        pass
    final_seis_ML[inde,:]=interp_x[-35064:]
    
st = UTCDateTime(2018, 1, 1)
et = UTCDateTime(2022, 1, 1)
windw = 1  # Time window in hours

# Generate the time array as numpy.datetime64
time_frame1 = []
current_time = st
while current_time < et:
    time_frame1.append(np.datetime64(current_time.datetime))  # Convert to numpy.datetime64
    current_time += 3600 * windw 

  
data_array = xr.DataArray(
    final_seis_ML,
    dims=("station", "time"),
    coords={"station": stations, "time": time_frame1},
    name="seismic_data"
)    
    
    
os.chdir("/Users/sebinjohn/ML_proj/Data")

data_array.to_netcdf("seismic_ML_train.nc")



fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(12,20))
for i in range(15):
    fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(16,10))
    axes.scatter(np.arange(0,len(res[i,:])),res[i,:],s=1)
    axes.set_ylim([-160,-90])    
    