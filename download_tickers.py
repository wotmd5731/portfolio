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
RTY=F	E-mini Russell 2000 Index Futur
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

SCZ
VTI
TLT
IEI
GLD
GSG
DBC

VWO (개발도상국 주식), 
BND (미국 총채권)


^TYX	Treasury Yield 30 Years	2.6030	+0.0920	+3.66%		
^TNX	Treasury Yield 10 Years	2.4920	+0.1510	+6.45%		
^IRX	13 Week Treasury Bill	0.5200	+0.0370	+7.66%		
^FVX	Treasury Yield 5 Years	2.5730	+0.2000	+8.43%		

^GSPC	S&P 500	4,543.06	+22.90	+0.51%	2.001B			
^DJI	Dow 30	34,861.24	+153.30	+0.44%	287.068M			
^IXIC	Nasdaq	14,169.30	-22.54	-0.16%	5.019B			
^NYA	NYSE COMPOSITE (DJ)	16,792.80	+90.98	+0.54%	0			
^XAX	NYSE AMEX COMPOSITE INDEX	4,264.68	+140.06	+3.40%	0			
^BUK100P	Cboe UK 100	744.91	+0.53	+0.07%	0			
^RUT	Russell 2000	2,077.98	+2.54	+0.12%	0			
^VIX	CBOE Volatility Index	20.81	-0.86	-3.97%	0			
^FTSE	FTSE 100	7,483.35	+15.97	+0.21%	0			
^GDAXI	DAX PERFORMANCE-INDEX	14,305.76	+31.97	+0.22%	0			
^FCHI	CAC 40	6,553.68	-2.09	-0.03%	0			
^STOXX50E	ESTX 50 PR.EUR	3,867.73	+4.34	+0.11%	0			
^N100	Euronext 100 Index	1,261.96	+1.90	+0.15%	0			
^BFX	BEL 20	4,119.42	+5.30	+0.13%	0			
IMOEX.ME	MOEX Russia Index	2,484.13	-94.38	-3.66%	0			
^N225	Nikkei 225	28,149.84	+39.45	+0.14%	0			
^HSI	HANG SENG INDEX	21,404.88	-541.07	-2.47%	0			
000001.SS	SSE Composite Index	3,212.24	-38.02	-1.17%	3.937B			
399001.SZ	Shenzhen Component	12,072.73	-232.78	-1.89%	3.556B			
^STI	STI Index	3,413.69	+13.99	+0.41%	0			
^AXJO	S&P/ASX 200	7,406.20	+19.10	+0.26%	0			
^AORD	ALL ORDINARIES	7,689.90	+20.90	+0.27%	0			
^BSESN	S&P BSE SENSEX	57,362.20	-233.48	-0.41%	0			
^JKSE	Jakarta Composite Index	7,002.53	-47.15	-0.67%	0			
^KLSE	FTSE Bursa Malaysia KLCI	1,603.30	+4.33	+0.27%	0			
^NZ50	S&P/NZX 50 INDEX GROSS	12,055.00	+37.39	+0.31%	0			
^KS11	KOSPI Composite Index	2,729.98	+0.32	+0.01%	602,771			
^TWII	TSEC weighted index	17,676.95	-22.11	-0.12%	0			
^GSPTSE	S&P/TSX Composite index	22,005.94	+68.05	+0.31%	255.478M			
^BVSP	IBOVESPA	119,081.13	+28.23	+0.02%	0			
^MXX	IPC MEXICO	55,436.05	-393.81	-0.71%	145.207M			
^IPSA	S&P/CLX IPSA	5,058.88	0.00	0.00%	0			
^MERV	MERVAL	38,390.84	+233.89	+0.61%	0			
^TA125.TA	TA-125	2,112.89	+8.50	+0.40%	0			
^CASE30	EGX 30 Price Return Index	11,709.20	-34.70	-0.30%	123.841M			
^JN0U.JO	Top 40 USD Net TRI Index	4,979.77	-23.69	-0.47%	0			


EURUSD=X	EUR/USD	1.0987	-0.0015	-0.13%		
JPY=X	USD/JPY	122.0600	-0.2600	-0.21%		
GBPUSD=X	GBP/USD	1.3189	+0.0003	+0.02%		
AUDUSD=X	AUD/USD	0.7516	+0.0004	+0.05%		
NZDUSD=X	NZD/USD	0.6975	+0.0011	+0.15%		
EURJPY=X	EUR/JPY	134.0400	-0.5000	-0.37%		
GBPJPY=X	GBP/JPY	161.0120	-0.2650	-0.16%		
EURGBP=X	EUR/GBP	0.8326	-0.0012	-0.14%		
EURCAD=X	EUR/CAD	1.3696	-0.0074	-0.54%		
EURSEK=X	EUR/SEK	10.3407	-0.0093	-0.09%		
EURCHF=X	EUR/CHF	1.0218	-0.0010	-0.10%		
EURHUF=X	EUR/HUF	372.1800	-1.8200	-0.49%		
EURJPY=X	EUR/JPY	134.0400	-0.5000	-0.37%		
CNY=X	USD/CNY	6.3658	-0.0012	-0.02%		
HKD=X	USD/HKD	7.8281	+0.0045	+0.06%		
SGD=X	USD/SGD	1.3575	+0.0002	+0.01%		
INR=X	USD/INR	76.2500	-0.0500	-0.07%		
MXN=X	USD/MXN	20.0200	-0.0500	-0.25%		
PHP=X	USD/PHP	52.1000	-0.1500	-0.29%		
IDR=X	USD/IDR	14,338.0000	-9.0000	-0.06%		
THB=X	USD/THB	33.5800	+0.0600	+0.18%		
MYR=X	USD/MYR	4.2080	-0.0160	-0.38%		
ZAR=X	USD/ZAR	14.5450	+0.0374	+0.26%		
RUB=X	USD/RUB	101.2500	+0.5000	+0.50%		


"""

aa= alltick.split('\n')
a_new=[]
for a in aa:
    if a.__len__()!=0:
        a_new.append(a.split()[0])
        
df = yf.download(a_new,period='max')
df.to_csv('alldata.csv')

#with open('alldata.dat','wb') as f:
#    pickle.dump(df,f)


# df = df['Adj Close']
