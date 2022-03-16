# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:58:08 2022

@author: JAE
"""

import pandas as pd
import numpy as np

import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint


np.set_printoptions(precision=2)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import seaborn as sns
import xlwings as xw


df = pd.read_pickle('alldata.dat')['Adj Close']

aa = df[['SPY','QQQ','UPRO','TLT','TMF']]

logr= np.log(1+aa.pct_change())
# logr.plot.hist()
ret=logr.mean()*252
std=logr.std()*252**0.5
norm_aa =  aa/aa.iloc[0]

norm_aa.rolling(22).mean().plot()

logr.rolling(22).mean().plot()
logr.rolling(6*22).mean().plot()
logr.rolling(1*22).std().plot()

aa.cov()

logr.cov()*252

a0 = aa[['SPY','TLT']].dropna()
a0.plot()

a1 = np.log(1+a0.pct_change())
sns.pairplot(logr)



#%%
df = pd.read_pickle('alldata.dat')
df = df['Adj Close']

col = df.columns[0]
df.columns
ll=[]
for col in df.columns:
    
    dd= df[col].dropna()
    if dd.__len__() <=0 :
        continue
    print("{} : {:%Y-%m-%d} ~ {:%Y-%m-%d}".format(dd.name, dd.index[0], dd.index[-1]))
    ll.append([dd.name,dd.index[0],dd.index[-1]])
    
    
df2 = pd.DataFrame(ll).sort_values(1)
df2.to_clipboard()



