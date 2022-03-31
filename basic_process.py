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
d = ['VXX', 'VXZ']
df = df.drop(columns=d)


dd =df['GLD SPY TLT'.split()]
dd = dd.dropna()
#%%
"""
추세 추종 매매. 모멘텀, 이런 이론과 방법론으로 돈을 벌수 잇다는 것은.
경제의 특수상황.
경제 위기. 인플레이션.경제 회복. 등등등 
다양한 상황에서  이러한  상태가 장기간 유지가 됨.

코로나 같은 위기로 급락이 발생한 직후.
2년에 걸쳐 장시간 증시가 계속 상승햇음. 
동일하게 금 또한 상승햇음.
위기가 오는 순간 채권이 상승햇음.

경재의 순환 사이클이라고 해야되나.
채권 상승 -> 주식 상승 ->경기상승-> 원자재 상승 -> 금리 인상 

주요 자산
주식, 채권, 원자재, TIPS

"""



df = yf.download('^VIX ^GSPC ^IXIC ^RUT CL=F GC=F ',period='max')
df = yf.download('^FVX ^TNX ^TYX ^IRX',period='max')
df = yf.download("ZN=F  ZT=F",period='max')

#%%
ddd = """
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



#%%
ddd = [dd.split()[0] for dd in ddd.split('\n') if dd.__len__() >0]
df = yf.download(ddd,period='max')
df = df['Adj Close']
df = df.fillna(method='ffill')
dd = df.dropna()
dd = dd/dd.iloc[0]
dd.plot()

#%%
d2 = dd[dd.index>"2019-01-01"]
d2= d2/d2.iloc[0]
d3 = 1/d2['EURUSD=X GBPUSD=X AUDUSD=X NZDUSD=X '.split()]
pd.concat([d3,d2['JPY=X']],axis=1).plot()

d2["CNY=X HKD=X SGD=X INR=X MXN=X PHP=X IDR=X THB=X MYR=X ZAR=X RUB=X".split()].plot()


#%%

ddd = """
^TYX	Treasury Yield 30 Years	2.6030	+0.0920	+3.66%		
^TNX	Treasury Yield 10 Years	2.4920	+0.1510	+6.45%		
^IRX	13 Week Treasury Bill	0.5200	+0.0370	+7.66%		
^FVX	Treasury Yield 5 Years	2.5730	+0.2000	+8.43%		
"""
ddd = [dd.split()[0] for dd in ddd.split('\n') if dd.__len__() >0]
df = yf.download(ddd,period='max')
df = df['Adj Close']
#%%

from mpl_toolkits import mplot3d
fig = plt.figure()
# ax = plt.axes(projection='3d')

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');


#%%

dd = dd/dd.iloc[0]
dd = np.log(dd)
dd.plot()
dd.cov()
dd.rolling(44).cov().unstack()['SPY'].plot()


c = df.columns[0]
cc=[]
for c in df.columns:
    dd = df[c].dropna()
    print("{} : {} ~ {} ".format(dd.name,dd.index[0],dd.index[-1]))
    cc.append([dd.name,dd.index[0],dd.index[-1]])

    
da = pd.DataFrame(cc)
da.sort_values(1)

#%%

d1 = df.dropna()
d2 = d1/d1.iloc[0]

dd = d2.cov()['SPY']
dd.to_clipboard()


d3 = np.log(d2)
dd = d3.cov()['SPY']
dd.to_clipboard()
aa = d3.rolling(22).cov().unstack()['SPY']


#%%
# aa = df[['VWO','HYG','SPY']].dropna(how='any')
# aa =  aa/aa.iloc[0]
# aa.to_clipboard()
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
