#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:19:22 2024

@author: sebinjohn
"""
import numpy as np
from obspy import UTCDateTime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pygmt
from global_land_mask import globe
import pandas as pd
import glob
import geopy.distance as dist
import xarray as xr
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
from joblib import load

seis=xr.open_dataset("/Users/sebinjohn/ML_proj/Data/seismic_ML_train.nc")
sea_ice = xr.open_dataset("/Users/sebinjohn/AON_PROJECT/Data/sea_ice_con/sea_ice_2018-2021-time.nc")
wave=xr.open_dataset("/Users/sebinjohn/AON_PROJECT/Data/wave/wave_2018-2021.nc")


wave.wave.data[np.isnan(wave.wave.data)]=0

df1=pd.read_csv("/Users/sebinjohn/AON_PROJECT/Data/station Meta Data.csv")


stations=seis.station.data

xw,yw=np.meshgrid(wave.lon,wave.lat)

fig=pygmt.Figure()
fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
fig.plot(x=xw.flatten(),y=yw.flatten(),style="c0.05c")
fig.plot(x=-156.61752020,y=71.3221,style="i0.2c",fill="blue")
fig.show() 

xw,yw=np.meshgrid(sea_ice.lon,sea_ice.lat)

fig=pygmt.Figure()
fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
fig.plot(x=xw.flatten(),y=yw.flatten(),style="c0.05c")
fig.plot(x=-156.61752020,y=71.3221,style="i0.2c",fill="blue")
fig.show() 


def thresh_cutter(distan,lon,lat,data,center_point):
    lon.data[lon.data>180]=lon.data[lon.data>180]-360
    x,y=np.meshgrid(lon,lat)
    boole1=globe.is_ocean(y,x)
    datao=data[boole1]
    lono=x[boole1]
    lato=y[boole1]
    points=[(ki, ka) for ki, ka in zip(lato, lono)]
    dist_arr=np.zeros((len(points)))
    for i in range(len(points)):
        dist_arr[i]=dist.geodesic(center_point,points[i]).km
    boole2=dist_arr<distan
    final_lat=lato[boole2]
    final_lon=lono[boole2]
    final_dat=datao[boole2]
    return final_lat,final_lon,final_dat

for i in range(1):
    i=-6
    stat=stations[i]
    lat_sta=df1[df1["Station\xa0"]==stat]["Latitude\xa0"]
    lon_sta=df1[df1["Station\xa0"]==stat]["Longitude\xa0"]
    wave_lat,wave_lon,wave_data=thresh_cutter(3000,wave.lon,wave.lat,wave.wave.data,(lat_sta.iloc[0],lon_sta.iloc[0]))
    sea_ice_lat,sea_ice_lon,sea_ice_data=thresh_cutter(3000,sea_ice.lon,sea_ice.lat,sea_ice.sea_ice_ml.data,(lat_sta.iloc[0],lon_sta.iloc[0]))
    fig=pygmt.Figure()
    fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
    fig.plot(x=wave_lon,y=wave_lat,style="c0.05c")
    fig.plot(x=-156.61752020,y=71.3221,style="i0.2c",fill="blue")
    fig.show()
    fig=pygmt.Figure()
    fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
    fig.plot(x=sea_ice_lon,y=sea_ice_lat,style="c0.05c")
    fig.plot(x=-156.61752020,y=71.3221,style="i0.2c",fill="blue")
    fig.show()
    
    ttl=len(wave_lat)+len(sea_ice_lat)
    #ttl=2
    X=np.zeros((5844,ttl))
    for ik in tqdm(range(5844)):
        X[ik,:]=np.hstack((sea_ice_data[:,ik].flatten(),wave_data[:,ik].flatten()))

    seis_ML=np.zeros((5844))
    j=-1
    for kk in tqdm(range(35064)):
        if kk%6==0:
            j=j+1
            seis_ML[j]=seis.seismic_data[i,kk]
    seis_ML[seis_ML==0]=np.nan  
    
    feature_rank=np.zeros(X.shape[1])
    avail_dat=[]
    for m in range(5844):
        if (np.isnan(seis_ML[m])):
            pass
        else:
            avail_dat.append(m)
    X1=X[avail_dat[:],:]
    y=seis_ML[avail_dat[:]]
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 20)
    #Scale data, otherwise model will fail.
    #Standardize features by removing the mean and scaling to unit variance
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    #Random forest.
    #Increase number of tress and see the effect
    from sklearn.ensemble import RandomForestRegressor
    n_estimators = 100
    model = RandomForestRegressor(n_estimators = 1, random_state=30,min_samples_split=6,oob_score=False,criterion="squared_error",warm_start=True)
    mses=[]
    with tqdm(total=n_estimators) as progress_bar:
        for i in range(n_estimators):
            model.n_estimators = i + 1  # Increase the number of trees by one
            model.fit(X_train_scaled, y_train)  # Train the new tree
            y_pred_RF = model.predict(X_test_scaled)
            mse_RF = mean_squared_error(y_test, y_pred_RF)
            mae_RF = mean_absolute_error(y_test, y_pred_RF)
            print('Mean squared error using Random Forest: ', mse_RF)
            print('Mean absolute error Using Random Forest: ', mae_RF)
            mses.append(mse_RF)
            progress_bar.update(1)  
    model_path = "/Users/sebinjohn/ML_proj/codes/RF/random_forest_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    mses_df = pd.DataFrame(mses, columns=["MSE"])
    mses_df.to_csv("/Users/sebinjohn/ML_proj/codes/RF/mses_list.csv", index=False)
    feature_i = np.array(model.feature_importances_)
    model = load(model_path)
    
    
    st=UTCDateTime(2018,1,1)
    et=UTCDateTime(2022,1,1)
    windw=6
    time_frame=np.array([])
    for i in range (int((et-st)/(3600*windw))):
        time_frame=np.append(time_frame,st)
        st=st+(3600*windw)
    
    plot_time=[]
    for ele in time_frame:
        plot_time.append(ele.matplotlib_date)    

    X_scaled = scaler.transform(X)
    y_full = model.predict(X_scaled)
    sub_time=[]
    for i in avail_dat:
        sub_time.append(plot_time[i])

    date_format=mdates.DateFormatter('%b')
    fig,ax=plt.subplots(figsize=(12,4))
    color = (0.082, 0.329, 0.396)
    ax.plot(sub_time,y,label="Observed SPSM power",c="k")
    ax.plot(plot_time,y_full,label="Predicted SPSM power",c="yellowgreen")
    ax.legend(loc="upper left")
    ax.set_xlim(min(plot_time),max(plot_time))
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.savefig("/Users/sebinjohn/AON_PROJECT/Results/RF_plots 2pixel and 100 km/3_pixel.png",transparent=True,dpi=300)
    





fig=pygmt.Figure()
fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
pygmt.makecpt(cmap="rainbow", series=[np.amin(feature_i[:sea_ice_lon.shape[0]]),np.amax(feature_i[:sea_ice_lon.shape[0]])],reverse=True)
fig.plot(x=sea_ice_lon,y=sea_ice_lat,style="c0.3c",cmap=True,fill=feature_i[:sea_ice_lon.shape[0]])
fig.plot(x=lon_sta,y=lat_sta,style="i0.5c",fill="blue")
fig.colorbar(frame=["a0.1f10", "x+lfeature importance"])
fig.show() 


fig=pygmt.Figure()
fig.coast(region=[170,230,50, 74], projection="L-159/35/33/45/22c",shorelines=False,frame="a",land="200/200/200",borders=["1/0.5p,black", "2/0.5p,red"])
pygmt.makecpt(cmap="rainbow", series=[np.amin(feature_i[sea_ice_lon.shape[0]:]),np.amax(feature_i[:sea_ice_lon.shape[0]:])],reverse=True)
fig.plot(x=wave_lon,y=wave_lat,style="c0.3c",cmap=True,fill=feature_i[sea_ice_lon.shape[0]:])
fig.plot(x=lon_sta,y=lat_sta,style="i0.5c",fill="blue")
fig.colorbar(frame=["a0.1f10", "x+lfeature importance"])
fig.show() 



sea_ice_data_mean=np.mean(sea_ice_data,axis=0)
wave_data_mean=np.mean(wave_data,axis=0)
spsm_data_mean=seis_ML

date_format=mdates.DateFormatter('%b')
fig,axe=plt.subplots(nrows=3,ncols=1,figsize=(12,8),sharex=True)
color = (0.082, 0.329, 0.396)
color1=(0.047, 0.533, 0.655)
axe[0].plot(plot_time,seis_ML,c="k",label="seismic_power",alpha=0.8)
axe[1].plot(plot_time,sea_ice_data_mean,c=color,label="sea_ice_concentration")
axe[2].plot(plot_time,wave_data_mean,c="blue",label="wave_height",alpha=0.8)
for ax in axe.flat:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(loc="upper left")
    ax.set_xlim(min(plot_time),max(plot_time))
fig.savefig("/Users/sebinjohn/AON_PROJECT/Results/pixel_close_power spsm/1.png",transparent=True,dpi=300)
