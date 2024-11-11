#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:56:47 2024

@author: sebinjohn
"""

import numpy as np
import matplotlib.pyplot as plt
from global_land_mask import globe
import pygmt
from scipy.interpolate import RegularGridInterpolator
import os
import pygrib 
import xarray as xr
import cdsapi
from tqdm import tqdm
from obspy import UTCDateTime

st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1
time_frame=np.array([])
for i in range (int((et-st)/(3600*windw))):
    time_frame=np.append(time_frame,st)
    st=st+(3600*windw)
    
def wave(tim):
    c = cdsapi.Client()
    os.chdir("/Users/sebinjohn/AON_PROJECT/Data/wave")
    files=os.listdir()
    if str(tim)[0:14]+"00"+".grib" not in files:
        print(str(tim)[0:14]+"00"+".grib not found Downloading...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 'significant_height_of_combined_wind_waves_and_swell',
                'year': str(tim.year),
                'month': str(tim.month),
                'day': str(tim.day),
                'time': tim.ctime()[11:14]+"00",
                'format': 'grib',
                },
            str(tim)[0:14]+"00"+".grib")
    tim_wav=str(tim)[0:14]+"00"
    grib=str(tim)[0:14]+"00"+".grib"
    grbs=pygrib.open(grib)
    grb=grbs[1]
    data_wave = grb.values
    latg, long = grb.latlons()
    lat=(latg[:,0])
    lon=(long[0,:])
    grid = xr.DataArray(
        data_wave, dims=["lat", "lon"], coords={"lat": lat, "lon": lon}
        )
    return grid,data_wave


windw=6

def wave_ak(time_frame,windw):
    for i in tqdm(range(len(time_frame))):
        if i%windw==0:
            grid,data_wave=wave(time_frame[i])
            ilat1=np.where(grid.lat==30)[0][0]
            ilat2=np.where(grid.lat==80)[0][0]
            ilon1=np.where(grid.lon==150)[0][0]
            ilon2=np.where(grid.lon==260)[0][0]
            wave_a=grid[ilat2:ilat1,ilon1:ilon2]
            if i==0:
                wave_ak=np.zeros((wave_a.shape[0],wave_a.shape[1],int(len(time_frame)/windw)))                
            wave_ak[:,:,int(i/windw)]=wave_a
    st=UTCDateTime(2018,1,1)
    et=UTCDateTime(2022,1,1)
    time_frame=np.array([])
    for i in range (int((et-st)/(3600*windw))):
        time_frame=np.append(time_frame,st)
        st=st+(3600*windw)
    plo_time=[]
    for ele in time_frame:
        plo_time.append(ele.matplotlib_date)
    data_array = xr.DataArray(wave_ak,dims=("lat","lon","time"),coords={"lat":grid.lat[ilat2:ilat1],
                "lon":grid.lon[ilon1:ilon2],
                "time":plo_time} )
    return data_array


data_array=wave_ak(time_frame,windw)
data_array.to_netcdf("/Users/sebinjohn/ML_proj/Data/wave_2018-2021.nc")



