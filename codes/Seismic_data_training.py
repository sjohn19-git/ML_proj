#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:14:52 2024

@author: sebinjohn
"""

import os
os.chdir("/Users/sebinjohn/AON_PROJECT/Data/codes")
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from obspy import UTCDateTime
from medfilt import medfilt



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
for i in range (95):
    freq.append(name.iloc[i]['freq'])
freq.append(19.740300000000000)
len(freq)