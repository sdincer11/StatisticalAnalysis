from numpy import *
import numpy as np
import statsmodels.api as sm
import pandas as pd
import sqlite3 as sql
import scipy.io as sio
from scipy import stats
import pickle
import os
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import subprocess
import copy
from collections import OrderedDict
import re
import time
import datetime
from scipy.stats import mstats
miktexPath='C:/Users/sdincer/AppData/Local/Programs/MiKTeX 2.9/miktex/bin/x64/'
projectPath='D:/Sinan Research Projects/Delta Hedged Options and Limits to Arbitrage/'
matricesPath=projectPath+'Matrices/'
csvPath=projectPath+'CSV Files/'
latexPath=projectPath+'Latex/'
latexTablesPath=latexPath+'tables - proposal 22 Oct 2020/'
figuresPath=latexPath+'figures/'


def get_winsorize(struct,winsorized_variables,pct=99,trim=False,byTime=False):
    for variable in winsorized_variables:
        var=struct[variable]
        if not byTime:
            critical_val_top=nanpercentile(var,pct)
            critical_val_bottom=nanpercentile(var,100-pct)
            indices_top=where(var>critical_val_top)
            indices_bottom=where(var<critical_val_bottom)
            if not trim:
                var[indices_top[0],indices_top[1]]=critical_val_top
                var[indices_bottom[0], indices_bottom[1]] = critical_val_bottom
            else:
                var[indices_top[0],indices_top[1]]=nan
                var[indices_bottom[0], indices_bottom[1]] = nan
            struct[variable]=var
        else:
            T,N=var.shape
            for t in range(0,T):
                critical_val_top=nanpercentile(var[t,:],pct)
                critical_val_bottom=nanpercentile(var[t,:],100-pct)
                indices_top=where(var[t,:]>critical_val_top)
                indices_bottom=where(var[t,:]<critical_val_bottom)
                if not trim:
                    var[t,indices_top[0]]=critical_val_top
                    var[t,indices_bottom[0]] = critical_val_bottom
                else:
                    var[t,:][indices_top[0],indices_top[1]]=nan
                    var[t,:][indices_bottom[0], indices_bottom[1]] = nan
            struct[variable]=var
    return struct

def read_pickles(columns):
    tables=dict()
    for column in columns:
        try:
            with open(''.join([matricesPath,column,'.txt']),"rb") as file:  # Pickling
                tables[column]=pickle.load(file)

        except Exception as e:
            print(e)
    return tables

def write_pickle(pickleFile,pickleData):

    with open(pickleFile,"wb") as file:  # Pickling
        pickle.dump(pickleData, file)

def set_dummy(matrix,condition):
    '''This function receives a Numpy array as the input as well as the condition in the string format,
     and returns a dummy numpy array as the output'''
    T,N=matrix.shape
    rank_dummy_matrix=full([T,N],nan)

    non_missing_code='where(~isnan(matrix))'
    non_missing_indices=eval(non_missing_code)
    non_missing_indices_x=non_missing_indices[0]
    non_missing_indices_y=non_missing_indices[1]
    rank_dummy_matrix[non_missing_indices_x, non_missing_indices_y] = 0

    code='where(matrix'+condition+')'
    indices=eval(code)
    indices_x=indices[0]
    indices_y=indices[1]
    rank_dummy_matrix[indices_x,indices_y]=1
    return rank_dummy_matrix

def set_portfolio_rank(variable,K,tstart):
    T,N = variable.shape
    ranked=full([T,N],nan)
    bounds=full([T,K+1],nan)
    for t in range(tstart,T):
        q = nanpercentile(variable[t, :], linspace(0, 100, K + 1))
        for i,el in enumerate(q[:-1]):
            if q[i]==q[i+1]:
                q[i+1]=q[i + 1] + 1

        bounds[t,:]=q
        for k in range(0,K):
            if k < K - 1:
                a = where((variable[t, :] >= q[k]) & (variable[t, :] < q[k + 1]))[0]
            elif k == K - 1:
                a = where((variable[t, :] >= q[k]) & (variable[t, :] <= q[k + 1]))[0]
            ranked[t,a]=k+1

    return ranked

def set_project_path(folderPath):
    global miktexPath,latexPath,latexTablesPath,projectPath,matricesPath,csvPath
    projectPath=folderPath
    matricesPath=projectPath+'Matrices/'
    csvPath=projectPath+'CSV Files/'
    latexPath=projectPath+'Latex/'
    latexTablesPath=latexPath+'tables/'
    figuresPath=latexPath+'figures/'


def fama_macbeth_regression(data,y_name,x_names,fixed_effects=None,constant=True,newey_west_lags=None):
    output=dict()
    y_matrix=data[y_name]
    T,N=y_matrix.shape
    number_indep_vars=len(x_names)
    if constant:
        number_indep_vars+=1
    betas=full([T,number_indep_vars],nan)
    nobs=full([T,1],nan)
    adjusted_rsquare=full([T,1],nan)
    first_time=min(where(any(~isnan(y_matrix),axis=1))[0])
    for t in range(first_time,T):
        try:
            y = y_matrix[t,:].reshape(-1,1)
            x = data[x_names[0]][t,:].reshape(-1,1)
            for indep_var in x_names[1:]:
                x = hstack((x,data[indep_var][t,:].reshape(-1,1)))
            if fixed_effects is not None:
                fixed_effect_vars = data[fixed_effects[0]][t,:].reshape(-1,1)
                for i in range(1,len(fixed_effects)):
                    fixed_effect_vars=hstack((fixed_effect_vars,data[fixed_effects[i]][t,:].reshape(-1,1)))
                ols_result=nanols(y,x,fixed_effects=fixed_effect_vars)
            else:
                ols_result = nanols(y, x)
            betas[t,:]=ols_result['betas']
            nobs[t,:]=ols_result['nobs']
            adjusted_rsquare[t]=ols_result['rsquared_adj']
        except Exception as e:
            print(e)
    if newey_west_lags is None:
        beta_summary = nantstat(betas)
        output['betas'] = beta_summary['mean']
        output['tstats']=beta_summary['tstat']
    elif newey_west_lags >0:
        beta_summary,tstat_summary=newey_west(betas,newey_west_lags)
        output['betas']=beta_summary
        output['tstats']=tstat_summary
    output['rsquared_adj']=nantstat(adjusted_rsquare[:])['mean']
    output['nobs']=nobs
    output['betas_list']=betas

    return output
def newey_west(betas,lags=6,kernel='bartlett'):
    #Regress betas on a constant by correcting for autocorrelated errors. Unconditional mean of betas will be the coefficient estimate on the constant
    if betas.size>0:
        T,K=betas.shape
        beta_means=full([K,1],nan)
        beta_tstats=full([K,1],nan)

        for k in range(0,K):
            beta=betas[:,k]
            beta=beta.reshape(-1,1)
            if lags>0:
                newey_west_result=nanols(beta,full([T,1],1),constant=False,robustness='HAC',cov_keywords={'maxlags':lags,'kernel':kernel})
            elif lags==0:
                newey_west_result=nanols(beta,full([T,1],1),constant=False,robustness='HC0')
            beta_means[k]=newey_west_result['betas'][0]
            beta_tstats[k]=newey_west_result['tstats'][0]

    return beta_means,beta_tstats
def create_specifications(struct2,y,x_vars,tstart,tend,lags):
    struct=copy.deepcopy(struct2)
    dates=struct['DATE']
    start=where(dates==tstart)[0][0]
    end=where(dates==tend)[0][0]
    y_lag=lags[0]
    x_vars_lags=lags[1:]
    result=dict()
    result[y]=struct[y][start+y_lag:end+y_lag+1,:]
    traversed=set()
    for idx,x_var in enumerate(x_vars):
        interacts= x_var.split('*')
        if len(interacts)>1:
            interacts=[term.strip() for term in interacts]
            interacted=1
            for i,term in enumerate(interacts):
                if term not in traversed:
                    temp=struct[term][start+x_vars_lags[idx][i]:end+x_vars_lags[idx][i]+1,:]
                else:
                    temp=result[term]
                interacted=interacted*temp
            result[x_var]=interacted
            traversed.add(x_var)
        else:
            result[x_var]=struct[x_var][start+x_vars_lags[idx]:end+x_vars_lags[idx]+1,:]
            traversed.add(x_var)
    return result
def run_specifications(data_struct,y,specifications,names,latexNameLookup,tableName,nd=2,const=True):

    if const==True:
        names=['CONSTANT']+names
    fe_list = []
    for spec in specifications:
        if spec.get('fixed_effects') is not None:
            fe_list.extend(spec.get('fixed_effects'))
    fe_list=list(set(fe_list))
    number_coeffs = len(names)
    number_specs=len(specifications)
    all_coeffs=full([number_coeffs,number_specs],nan).astype(object)
    all_tstats = full([number_coeffs, number_specs], nan).astype(object)
    rsquared_adj=full([1,number_specs],nan)
    nobs=full([1,number_specs],nan)
    for spec_i,spec in enumerate(specifications):
        spec_vars=spec.get('variables')
        spec_fes=spec.get('fixed_effects')
        betas=full([number_coeffs,1],nan)
        tstats = full([number_coeffs, 1], nan)
        idx=where(isin(names,spec_vars))[0]
        if const == True:
            idx=insert(idx,0,0)
        fama_macbeth_result = fama_macbeth_regression(data_struct, y, spec_vars,fixed_effects=spec_fes,constant=const,newey_west_lags=None)
        betas[idx]=fama_macbeth_result['betas'].reshape(-1, 1) * 100
        tstats[idx]=fama_macbeth_result['tstats'].reshape(-1, 1)
        nobs[0,spec_i]=nanmean(fama_macbeth_result['nobs'],axis=0)
        rsquared_adj[0,spec_i]=fama_macbeth_result['rsquared_adj']
        all_coeffs[:,spec_i]=betas[:,0]
        all_tstats[:,spec_i]=tstats[:,0]

    for j,fe in enumerate(fe_list):
        for i,spec in enumerate(specifications):
            idx=where(isin(names,fe))[0]
            if spec.get('fixed_effects') is not None:
                if fe in spec.get('fixed_effects'):
                    all_coeffs[idx,i] ='Yes'
            else:
                all_coeffs[idx,i]='No'
    all_coeffs=np.append(all_coeffs,rsquared_adj,axis=0)
    all_tstats=np.append(all_tstats,full([1,number_specs],nan),axis=0)
    all_coeffs=np.append(all_coeffs,nobs,axis=0)
    all_tstats=np.append(all_tstats,full([1,number_specs],nan),axis=0)

    names=[latexNameLookup.get(name,'') for name in names]+['$Adjusted R^{2}$','Number of Observations(monthly average)']
    table_coef_tstatfixed(all_coeffs,all_tstats,names,2,tableName,addStars=True)
    return all_coeffs,all_tstats


def create_pdf(input_filename):
    process = subprocess.Popen([miktexPath+'pdflatex.exe',input_filename], cwd=latexPath)
    process.communicate()
    os.remove(latexPath + input_filename + '.log')
    os.remove(latexPath + input_filename + '.out')
    subprocess.Popen([latexPath+input_filename+'.pdf'],shell=True)
def event_study(returns,permnos,permnoIndices,eventWindow=12):
    eventReturns=full([len(permnoIndices),2*eventWindow+1],nan)
    T,N=returns.shape
    for record_idx,permno_index in enumerate(permnoIndices):
        try:
            if len(permno_index['PERMNO_IDX'])>0:
                t = permno_index['T']
                event_start=max(t-eventWindow,0)
                event_end=min(t+eventWindow,T-1)
                for timeIdx,event_t in enumerate(list(range(t,event_start-1,-1))):
                    eventReturns[record_idx,eventWindow-timeIdx]=nanmean(returns[event_t,permno_index['PERMNO_IDX']])
                    k=2
                for timeIdx,event_t in enumerate(list(range(t,event_end+1))):
                    eventReturns[record_idx,eventWindow+timeIdx]=nanmean(returns[event_t,permno_index['PERMNO_IDX']])
                    u=2
        except Exception as e:
            print(e)
    T,window=eventReturns.shape
    cum_returns=full([T,window],nan)
    tstart=eventWindow-1
    cum_returns[:,tstart]=0.0
    for t in range(0,T):
        nanmissing_idx=where(~isnan(eventReturns[t,:]))[0]
        if len(nanmissing_idx)>0:
            to_cumulate=[month for month in list(range(min(nanmissing_idx),max(nanmissing_idx)+1)) if month>tstart]
            for idx,ret_idx in enumerate(to_cumulate):
                if ~math.isnan(eventReturns[t,idx]):
                    if idx==0:
                        cum_returns[t,ret_idx]=eventReturns[t,ret_idx]
                    else:
                        cum_returns[t,ret_idx]=(1+eventReturns[t,ret_idx])*(1+cum_returns[t,ret_idx-1])-1
    cum=nanmean(cum_returns,axis=0)
    evt=nanmean(eventReturns,axis=0)
    return cum,evt





def lag(dF,variables,lags):

    for var_idx,var in enumerate(variables):
        dF_copy=dF[dF[var].notnull()]
        permnos=np.sort(dF_copy['PERMNO'].unique())
        dates=np.sort(dF_copy['YYYYMM'].unique())
        dF_copy.sort_values(by=['PERMNO','YYYYMM'],inplace=True)
        mat=dF_copy.pivot(index='YYYYMM',columns='PERMNO',values=var).as_matrix()
        mat_lag=np.roll(mat,lags[var_idx],axis=0)
        mat_lag[:lags[var_idx],:]=np.nan
        nanmissing=np.where(~np.isnan(mat_lag))
        permno_indices=nanmissing[1]
        date_indices=nanmissing[0]
        nanmissing_values=mat_lag[date_indices,permno_indices].reshape(-1,1)
        val=np.hstack((permnos[permno_indices].reshape(-1,1),dates[date_indices].reshape(-1,1),nanmissing_values))
        lagDF=pd.DataFrame(columns=['PERMNO','YYYYMM',var+'_LAG'+str(lags[var_idx])],data=val)
        dF=pd.merge(dF,lagDF,how='left',on=['PERMNO','YYYYMM'])

    return dF











