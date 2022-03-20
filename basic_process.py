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

import pyqtgraph as pg
# df = pd.read_pickle('alldata.dat')['Adj Close']
df = pd.read_csv('alldata.csv').set_index('Date')


#%%
aa = df[['VWO','HYG','SPY']].dropna(how='any')
aa =  aa/aa.iloc[0]
aa.to_clipboard()
#%%



aa = df[['SPY','TLT']]
# aa = df[['SPY']]

aa=aa.dropna()
aa =  aa/aa.iloc[0]

logr= np.log(1+aa.pct_change())
# logr.plot.hist()
ret=logr.mean()*252
std=logr.std()*252**0.5

#%%

# l0 = logr.rolling(22,min_periods=1).mean()*252
# l0.to_clipboard()


a_list= []
for idx in  range(aa.__len__()):
    if idx>45:
        a0 = (aa.iloc[idx] / aa.iloc[idx-44]-1)
        a_list.append(a0)
        
a1 = pd.concat(a_list,axis=1)

a1 = a1.transpose()    



#%%
aa['SPY'].plot()
aa['SPY'].to_clipboard()


pd.concat([aa,rstd],axis=1)

a2 = logr.rolling(44).cov().unstack()['SPY']['TLT']*252
a2.to_clipboard()

logr.rolling(22).mean().plot()
logr.rolling(6*22).mean().plot()
logr.rolling(2*22).std()*252**0.5


aa.cov()

logr.cov()*252

a0 = aa[['SPY','TLT']].dropna()
a0.plot()

a1 = np.log(1+a0.pct_change())
sns.pairplot(logr,kind='reg')



logr

#%%
aa = df[['Date','UPRO','TMF','SPY','TLT']]
aa = aa.set_index('Date')
aa
aa=aa.dropna()
aa =  aa/aa.iloc[0]
aa

logr= np.log(1+aa.pct_change())
#portfolio return  2method

# # 1 method 
# ll = logr*[0.6,0,0.4,0]
# l1 = ll.sum(1)
# np.exp(l1.cumsum())


# # 2 method
# ll = logr*[0.6,0,0.4,0]
# l2 = np.exp(ll.sum(1))
# rr=l2.cumprod()

# # 
# rdd = (rr - rr.cummax())/rr.cummax()
# rdd.min()

#%%
ww=[0.6,0,0,0.4]
def mean_var(logr,ww):
    l0 = logr*ww
    l1 = l0.sum(1)
    std = l1.std()*252**0.5
    ann_ret= l1.mean()*252
    ret = np.exp(l1.sum())
    l2 = np.exp(l1)
    rr=l2.cumprod()
    rdd = (rr - rr.cummax())/rr.cummax()
    print("AnnR:{:.3f} R:{:.3f} STD:{:.3f} DD:{:.3f}".format(ann_ret,ret,std,rdd.min()))
    


mean_var(logr,[0,0,0.6,0.4])
mean_var(logr,[0,0.4,0.6,0])

mean_var(logr,[0,0,1,0])
mean_var(logr,[0.6,0.4,0,0])
mean_var(logr,[1,0,0,0])

#%%

logr.cov()






#%%

ret=logr.mean()*252
std=logr.std()*252**0.5

ll = logr*[0.6,0.4,0,0]
ll.sum(1).mean()*252
ll = logr*[0.6,0,0.4,0]
l1 = ll.sum(1)
l1.cumsum()
np.exp(ll.sum(1).sum())



l2 = np.exp(ll.sum(1))
rr=l2.cumprod()
rdd = (rr - rr.cummax())/rr.cummax()
rdd.min()

aa['SPY'].plot()
logr


g = sns.pairplot(ab,kind='reg')
for ggaa in g.axes:
    for  gg in ggaa:
        print(gg)
        gg.set_xlim((-0.2,0.2))
        gg.set_ylim((-0.2,0.2))



logr.rolling(22).cov().unstack()['SPY'].to_clipboard()

ab =logr.rolling(22).cov().unstack()*252
ab.plot()
logr.rolling(22).corr().unstack()['SPY'].to_clipboard()

np.exp(logr.cov()*252)
logr.cov()*252


logr.rolling(44).std()['SPY'].to_clipboard()

#%%

bb = logr*[0.6,0,0.4]
bb.sum(1).mean()*252

bb = logr*[0.6,0.4,0]
bb.sum(1).mean()*252

(bb.mean()*252).sum()
b1 = bb.sum(1)
b1


np.exp(bb.sum(1).sum())
