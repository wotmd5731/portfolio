#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
SIXY   -1x
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


"""

# aa= alltick.split('\n')
# a_new=[]
# for a in aa:
#     if a.__len__()!=0:
#         a_new.append(a.split()[0])
        
# df = yf.download(a_new,period='max')

# df.to_pickle('alldata.dat')

# "SHY IEF TLT EDV VTIP TIP LTPZ "
 # US equities (represented by SPY), European equities (EZU), Japanese equities (EWJ), emerging market equities (EEM), US REITs (VNQ), international REITs (RWX), US 10-year Treasuries (IEF), US 30-year Treasuries (TLT), commodities (DBC) and gold (GLD).
"SPY EZU EWJ EEM VNQ RWX IEF TLT DBC GLD"
'SPY EFA EEM QQQ DBC GLD TLT IWM IYR'
# In[31]:
#%% BASE CLASS
#%% BASE CLASS
#%% BASE CLASS

class portfolio():
    def __init__(self):
        self.ticker=[]
        self.df=[]
        self.MIN_START_IDX=20
        self.REBAL_MONTH=1
        self.workingday=252
    
    
    
    def setTicker(self,ticker):
        self.ticker= ticker
    def getTicker(self):
        return self.ticker
    
    
    def download(self,ticker):
        df= yf.download(ticker,period='max')
        df= df['Adj Close']
        return df

    def saveDf(self,df):
        df.to_pickle('data.dat')
    def loadDf(self,load_all=False):
        if load_all==True:
            try:
                df = pd.read_pickle('alldata.dat')
                return df
            except:
                import pickle5
                
                with open('alldata.dat','rb') as ff:
                    df =pickle5.load(ff)
                    return df

        else:
            df=pd.read_pickle('data.dat')
            return df

    
    def toExcel(self,df):
        wb = xw.books[0]
        ws= wb.sheets[0]
        ws[0,0].value=df

    def dropInvalid(self,df):
        df = df.fillna(method='ffill')
        df = df.dropna(axis=0,how='any')
        return df

    def setDayInfo(self,df):
        df= df.assign(m_index=df.index.month)
        df= df.assign(m_end= abs(df.m_index.shift(-1)-df.m_index)>0)
        df= df.dropna()
        return df

    def setNormalize(self,df,ticker):
        df[ticker] = df[ticker]/df[ticker].iloc[0]
        return df
    
    
    def calcDD(self,df):
        dd= df/df.rolling(window=self.workingday,min_periods=1).max()-1
        return dd
    def calcMDD(self,df):
        return self.calcDD(df).min()
    def logReturn(self,df):
        return np.log(1+df.pct_change())
    def calcAnnualStd(self,df):
        return self.logReturn(df).std()*np.sqrt(self.workingday)
    def calcAnnualMean(self,df):
        return self.logReturn(df).mean()*self.workingday
    def calcCAGR(self,df):
        CAGR = ((df.iloc[-1]/df.iloc[0])**(1/df.iloc[0:-1].__len__()*self.workingday)-1)
        return CAGR
    def calcCAGR_recent(self,df):
        c_month=[1,6,12,24]
        str_month=[str(c)+'M'for c in c_month]
        c_cagr=[]
        for mon in c_month:
            day= df.index[-1] - relativedelta(months=mon)
            idx = np.abs(df.index-day).argmin()
            cagr = (df.iloc[-1]/df.iloc[idx])**(1/df.iloc[idx:-1].__len__()*self.workingday)-1
            c_cagr.append(cagr)
        dd= pd.concat(c_cagr,axis=1)
        dd.columns=str_month
        return  dd
    
    def report(self):
        d0=self.out_df[self.ticker+['P_ev']]
        rpt=[]
        rpt.append(self.calcAnnualMean(d0))
        rpt.append(self.calcAnnualStd(d0))
        rpt.append(self.calcCAGR(d0))
        rpt.append(self.calcMDD(d0))
        a= pd.concat(rpt,axis=1)
        a.columns=['AnnMean','AnnStd','CAGR','MDD']
        b = self.calcCAGR_recent(d0)
        d1= pd.concat([a,b],axis=1)
        #d1.to_clipboard()
        return d1
        
    def custom_init(self):
        self.bal['SPY']=0.6
        self.bal['TLT']=0.4
        
    def custom_rebal(self,idx):
        print(self.bal)
        pass
    
    
    def initDf(self):
        assert self.df.__len__() != 0
        df=self.df[self.ticker]
        df=self.dropInvalid(df)
        df=self.setDayInfo(df)
        self.df=self.setNormalize(df,self.ticker)
    def initPf(self):
        assert self.ticker.__len__()!=0
        
        self.df=self.loadDf(True)['Adj Close']       
        self.initDf()
        self.pf = pd.Series(dict.fromkeys(self.ticker,0)).astype(float)
        self.pf['ev']=0
        self.pf['cash']=1
        self.pf_list=[]
        self.bal=pd.Series(dict.fromkeys(self.ticker,0)).astype(float)

        
    def test_pf(self):
        self.initPf()
        """---------------custom init start---------------"""
        self.custom_init()
        """---------------custom init end---------------"""
        for idx in range(self.df.__len__()):
            cur_dd=self.df.iloc[idx]
            eval_stock = [self.pf[tt]*cur_dd[tt] for tt in self.ticker]
            self.pf['ev'] = sum(eval_stock)+self.pf['cash']
            print(idx,self.pf['ev'])
            self.pf_list.append(self.pf.copy())
            if idx>self.MIN_START_IDX and cur_dd.m_index % self.REBAL_MONTH ==0 and cur_dd.m_end == True:
                """---------------custom rebal start---------------"""
                self.custom_rebal(idx)
                """---------------custom rebal start---------------"""
                self.pf[self.ticker]=self.pf['ev']*self.bal[self.ticker]/cur_dd[self.ticker]
                self.pf['cash']=self.pf['ev']-(self.pf[self.ticker]*cur_dd[self.ticker]).sum()
        pfl=pd.DataFrame(self.pf_list)
        pfl.columns= "P_"+pfl.columns
        pfl.index = self.df[:pfl.__len__()].index
        self.out_df=pd.concat([self.df,pfl],axis=1)
       
pf = portfolio()
pf.setTicker('SPY TLT'.split())


#%%  equal weight

class pf_AAA(portfolio):
    def ret_risk(self,rnd_w, df_mean,  cov_mat):
        pf_ret = np.sum(df_mean*rnd_w)
        pf_vol=np.sqrt(rnd_w.T@cov_mat@rnd_w)
        pf_sharpe = pf_ret/pf_vol
        return pf_vol
        # return -pf_sharpe

    # W = np.ones((factor_moments.shape[0],1))*(1.0/factor_moments.shape[0])
    # Function that runs optimizer
    def optimize(self,func, rnd_w, df_mean, cov_mat):
        opt_bounds = Bounds(0, 1)
        opt_constraints = ({'type': 'eq',
                            'fun': lambda W: 1.0 - np.sum(W)})
        optimal_weights = minimize(func, rnd_w, 
                                   args=(df_mean, cov_mat),
                                   method='SLSQP',
                                   bounds=opt_bounds,
                                   constraints=opt_constraints)
        return optimal_weights['x']

    
    def __init__(self):
        super().__init__()
        pass
    
    def custom_init(self):
        self.n_top = 5
        self.n_mom = 6*22
        self.n_vol= 1*22

        pass
    def custom_rebal(self,idx):
        idx_mom = (idx-self.n_mom)if idx-self.n_mom>0 else 0 
        df_mom = np.log(1+self.df.iloc[idx_mom:idx+1][self.ticker].pct_change()).mean()
        asset = df_mom.nlargest(self.n_top)

        idx_vol= (idx-self.n_vol) if idx-self.n_vol>0 else 0
        df_vol= self.df.iloc[idx_vol:idx+1][asset.index].pct_change()
        df_mean = df_vol.mean()*self.workingday
        cov_mat= df_vol.cov()*self.workingday
        rnd_w = np.random.dirichlet(np.ones(asset.__len__()),size=1).reshape(-1)


        tar_w = self.optimize(self.ret_risk, rnd_w , df_mean ,cov_mat)
        print(tar_w, self.ret_risk(tar_w,df_mean,cov_mat))


        self.bal = pd.Series(dict.fromkeys(self.ticker,0)).astype(float)
        self.bal[asset.index.tolist()]=tar_w
        pass
    
pf2= pf_AAA()
pf2.setTicker('SPY EFA EEM QQQ DBC GLD TLT IWM IYR'.split())
#pf2.test_pf()
#pf2.report()


# In[16]:



#%%
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
