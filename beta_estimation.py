
# coding: utf-8

# ## Beta estimating

# In[8]:
    
from scipy.stats import mstats
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('trading_data/index_future_close.csv',index_col=0)
df = df.dropna()


# In[23]:

def ols_beta(df, window):
    model = pd.stats.ols.MovingOLS(y=df.ic, x=df[['ih']], window_type='rolling', window=window, intercept=True)
    df['ols_beta'] = model.beta.ih.shift(1)
    df['ols_r2'] = model.r2
    df.ols_beta = mstats.winsorize(df.ols_beta, limits=[0.01, 0.01])
    #df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = ols_beta(df, window = 5000)
#if 'ols_beta' not in df:
#    df.to_csv('trading_data/index_future_close.csv')


# In[11]:

from pykalman import KalmanFilter
initial_state_mean = 0.6
initial_state_covariance = 50

# There shouldn't be a constant in the measurement. Interception is the alpha we want to capture. 
kf = KalmanFilter(transition_matrices = [1],
                 observation_matrices = [0], 
                 initial_state_mean = initial_state_mean,
                 initial_state_covariance = initial_state_covariance,
                 observation_covariance = 1,
                 transition_covariance = 0.01)

n_timesteps,n_dim_state,n_dim_obs = len(df),1,1
filtered_state_means = np.zeros((n_timesteps, n_dim_state))
filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
filtered_state_means[0] = initial_state_mean
filtered_state_covariances[0] = initial_state_covariance

for i in tqdm(range(n_timesteps-1)):
    filtered_state_means[i+1], filtered_state_covariances[i+1] = kf.filter_update(filtered_state_means[i],
                                                                            filtered_state_covariances[i],
                                                                            df.ic[i],
                                                                            observation_matrix = df.ih[i])

df['kf_beta'] = filtered_state_means
state_cov = [x[0][0] for x in filtered_state_covariances]
df['state_cov'] = state_cov


# In[24]:
xticks = pd.to_datetime(df.index).date.astype(str)
df[['kf_beta','ols_beta']].plot(x=xticks,figsize=(10,6),rot=45,title='OLS and KF beta')
plt.figure()
df['state_cov'].plot(x=xticks, figsize=(10,6),rot=45)


# In[13]:

from math import *
df['if_cum'] = df['if'].cumsum()

def mse_of_beta(df, beta_col_name, plot=False):
    print('==== Beta column: {} ====='.format(beta_col_name))
    r2_thresh = 0
    if r2_thresh:
        ls = (df.ic-df[beta_col_name]*df.ih)*(df.r2>r2_thresh)
        print('Open position ratio:',len(df[df.r2>r2_thresh])/len(df))
    else:
        ls = df.ic-df[beta_col_name]*df.ih
        
    ls_cum = ls.cumsum()
    mse = (ls**2).mean()
    if plot:
        pd.DataFrame({'smb_cum':ls_cum, 'if_cum':df.if_cum}).plot()
    
    print('STD:', ls.std())
    print('MSE:', mse)
    print('SR:', ls.mean()/ls.std()*sqrt(250))
    print()
    return ls

df['kf_resid'] = mse_of_beta(df, 'kf_beta')
df['ols_resid'] = mse_of_beta(df, 'ols_beta')

df['constant_beta'] = 1
df['constant_resid'] = mse_of_beta(df, 'constant_beta')

# In[15]:


smb_all = df[['kf_resid','ols_resid','equal_weight_resid']].dropna()
xticks = pd.to_datetime(smb_all.index).date.astype(str)
smb_all.cumsum().plot(x=xticks, figsize=(10,6),rot=45)


# In[10]:
# Measurement with intercept.
def kf_measure_with_intercept():
    initial_state_mean = 0.8
    initial_state_covariance = 1
    
    # 构建一个卡尔曼滤波器
    kf = KalmanFilter(transition_matrices = np.eye(2),
                     observation_matrices = [0.01,1], 
                     initial_state_mean = [initial_state_mean, 0],
                     initial_state_covariance = np.ones((2, 2)),
                     observation_covariance = 1,
                     transition_covariance = np.eye(2) * 0.0001)
    
    n_timesteps,n_dim_state,n_dim_obs = len(df),2,1
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))
    filtered_state_means[0] = initial_state_mean
    filtered_state_covariances[0] = initial_state_covariance
    
    for i in tqdm(range(n_timesteps-1)):
        filtered_state_means[i+1], filtered_state_covariances[i+1] = kf.filter_update(filtered_state_means[i],
                                                                                filtered_state_covariances[i],
                                                                                df.ic[i],
                                                                                observation_matrix = np.array([[df.ih[i],1]]))
    
    #df['kf_beta'] = filtered_state_means

