#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:51:37 2024

@author: sebinjohn
"""

import numpy as np
import numpy.ma as ma
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
import pyproj 
import pygmt
import xarray as xr
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
from global_land_mask import globe
import geopy.distance as dist
from tqdm import tqdm
from obspy import UTCDateTime
from scipy.interpolate import interp1d
import matplotlib.dates as mdates


st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1*24
time_frame=np.array([])
for i in range (int((et-st)/(3600*windw))):
    time_frame=np.append(time_frame,st)
    st=st+(3600*windw)


sea_ice_con=np.load("/Users/sebinjohn/AON_PROJECT/Data/sea_ice_con/sea_ice_con.npy")

#sea_ice_con = ma.masked_greater(sea_ice_con, 1.0)
ice=sea_ice_con[:,:,0]

dx = dy = 25000

x = np.arange(-3850000, +3750000, +dx)
y = np.arange(+5850000, -5350000, -dy)

x.shape, y.shape, ice.shape


fig = plt.figure(figsize=(9, 9))
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))

cs = ax.coastlines(resolution='110m', linewidth=0.5)

ax.gridlines()
ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

kw = dict(central_latitude=90, central_longitude=-45, true_scale_latitude=70)
cs = ax.pcolormesh(x, y, ice, cmap=plt.cm.Blues,
                   transform=ccrs.Stereographic(**kw))



X,Y=np.meshgrid(x,y)

NorthPolar_WGS=pyproj.Transformer.from_crs(3413,4326)
WGSvalues=NorthPolar_WGS.transform(X,Y)

lat=WGSvalues[0]
lon=WGSvalues[1]

###re_gridding

boollat=np.logical_and(lat >= 60, lat <= 85)
boollon=np.logical_or(np.logical_and(lon >= -180, lon <= -140),np.logical_and(lon >= 150, lon <= 179.9))
boole=np.logical_and(boollat,boollon)

points=np.array((lon[boole],lat[boole].flatten())).T

x_s=np.arange(-180,-140,0.5)
y_s=np.arange(60,81,0.5)
grid_x, grid_y = np.meshgrid(x_s,y_s) 

boolei=globe.is_ocean(grid_y,grid_x)


sea_ice_ml=np.zeros((grid_x.shape[0],grid_x.shape[1],1461))
for i in tqdm(range(1461)):
    values=sea_ice_con[:,:,i][boole]
    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    sea_ice_ml[:,:,i]=grid_z0

sea_ice_ml[sea_ice_ml>1]=0

sea_ice_ml.shape

    
plo_time=[]
for ele in time_frame:
    plo_time.append(ele.matplotlib_date)


data_array = xr.DataArray(sea_ice_ml,
                    dims=("lat","lon","time"),
                    coords={"lat":y_s,
                            "lon":x_s,
                            "time":plo_time}
                    )
dataset = xr.Dataset({'sea_ice_ml': data_array})




######time_interpol



st=UTCDateTime(2018,1,1)
et=UTCDateTime(2022,1,1)
windw=1*6
time_frame=np.array([])
for i in range (int((et-st)/(3600*windw))):
    time_frame=np.append(time_frame,st)
    st=st+(3600*windw)
    

#ice_grdcut_ocean=np.load("sea_ice_grdcut.npy")
sea_ice_timeinter=np.zeros((data_array.shape[0],data_array.shape[1],len(time_frame)))

torg=data_array.time.data

interp_func = interp1d(torg, data_array, kind='linear',fill_value="extrapolate")

plo_time=[]
for ele in time_frame:
    plo_time.append(ele.matplotlib_date)

mdates.num2date(np.max(plo_time))
mdates.num2date(np.max(torg))

interp_values = interp_func(plo_time)

x = xr.DataArray(data=interp_values, dims=('lat', 'lon', 'time'), coords={'lat': y_s, 'lon': x_s, 'time':plo_time })

dataset1 = xr.Dataset({'sea_ice_ml': x})

dataset1.to_netcdf("/Users/sebinjohn/ML_proj/Data/sea_ice_2018-2021-ml.nc")

