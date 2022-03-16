# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:48:10 2022

@author: JAE
"""

import pandas as pd
import numpy as np

import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

import pickle
np.set_printoptions(precision=2)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import seaborn as sns
import xlwings as xw
#%%
alltick="""
AGG US Tbond aggregate 
BND total bond
BNDX total international bond
BSV 단기 세계

LQD 투자등급 회사채

MBB  MBS 채권
IGSB 1-5Y 투자등급 회사채


VCIT 중기 회사채
VCSH 단기 회사채


HYG  high yield 회사채
JNK  high yield 회사채
SRLN  시니어 론 ETF 



SHV  1M~1Y Tbond
SHY  1~3Y Tbond  
IEF  7~10Y Tbond
TLT  19Y Tbond
VGLT 18Y Tbond
EDV  25Y Tbond

SCHP 7.5Y TIPS
VTIP 1~3Y 단기 TIPS
TIP  7~10 중기 TIPS
LTPZ   20+ 장기 TIPS

UST  중기 2x
TYD  중기 3x
UBT  장기 2x
TMF  장기 3x

TBX  중기 -1x
PST  중기 -2x

TMV  장기 -3x

TBF  장기 -1x
TBT  장기 -2x
TTT  장기 -3x

SH   sp500 -1x
SDS  sp500 -2x
SPXU sp500  -3x
SPXS  sp500 -3x #거래량 적음
SPY
SSO    2x
SPXL    3x
UPRO  3x

UVXY  x1.5 vix 단기 
TVIX   x2  ETN
VIXY  vix 단기

VIXM  중기 vix
SVXY  vix 단기 -1x
VXX   vix 단기  ETN 최초 상품
VXZ  중기 vix ETN



QQQ   1x
PSQ  -1x
QID  -2x

SQQQ  -3x
TQQQ  3x
QLD x2

QYLD  qqq커버드 콜


GLD
SLV

SPY 
EFA 
EEM 
QQQ
DBC
GLD 
TLT 
IWM 
IYR

EZU
EWJ
VNQ
RWX
IEF

ES=F 
NQ=F 
ZB=F  U.S. Treasury Bond
ZN=F  10-Year T-Note Futures
ZF=F  Five-Year US Treasury
ZT=F  2-Year T-Note Futures
GC=F 
SI=F 
HG=F 
CL=F

UCO   2x crude oil
SCO -2x
BOIL  gas 2x
KOLD      ---2x
UGL   Gold 2x
GLL       2x
AGQ   silver 2x
ZSL        -2x




"""

aa= alltick.split('\n')
a_new=[]
for a in aa:
    if a.__len__()!=0:
        a_new.append(a.split()[0])
        
df = yf.download(a_new,period='max')

with open('alldata.dat','wb') as f:
    pickle.dump(df,f)


# df = df['Adj Close']
