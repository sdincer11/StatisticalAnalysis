from numpy import *
import numpy as np
from NumpyRegressions import *
def get_risk_adjusted_portfolio_returns(returns,factors,constant=True,robust_type='HC0'):
    'This function adjusts the portfolio returns by risk factors'

    dims = list(returns.shape)
    adjusted_returns = full(dims, nan)

    if len(dims)==4:
        K3=dims[3]

    dims=dims[1:]
    K1=dims[0]
    if len(dims)>1:
        K2=dims[1]
    adjusted_return_mean=full(dims,nan)
    adjusted_return_tstats = full(dims, nan)
    rf=factors[:,0:1]
    factors=factors[:,1:]

    if len(dims)==1:
        for k1 in range(0,K1):
            try:
                if k1!=K1-1:
                    excess_return=returns[:,k1].reshape(-1, 1)-rf
                else:
                    excess_return=returns[:,k1].reshape(-1, 1)
                ols_results=nanols(excess_return,factors,constant)
                if constant==True:
                    adjusted_returns[:,k1]=(ols_results['resids']+ols_results['betas'][0])[:,0]
                    summary = nantstat(adjusted_returns[:,k1])
                    adjusted_return_mean[k1]=summary['mean']
                    adjusted_return_tstats[k1]=summary['tstat']
                else:
                    adjusted_returns[:,k1] = excess_return[:, 0] - dot(factors,ols_results['betas'].reshape(-1,1))[:,0]
                    summary=nantstat(adjusted_returns)
                    adjusted_return_mean[k1]=summary['mean']
                    adjusted_return_tstats[k1] = summary['tstat']
            except Exception as e:
                    print(e)

    elif len(dims)==2:

        for k1 in range(0,K1):
            for k2 in range(0,K2):
                try:
                    if k1!=K1-1 and k2!=K2-1:
                        excess_return=returns[:,k1,k2].reshape(-1, 1)-rf
                    else:
                        excess_return=returns[:,k1,k2].reshape(-1, 1)
                    ols_results=nanols(excess_return,factors,constant)
                    if constant==True:
                        adjusted_returns[:,k1,k2]=(ols_results['resids']+ols_results['betas'][0])[:,0]
                        summary = nantstat(adjusted_returns[:,k1,k2])
                        adjusted_return_mean[k1,k2]=summary['mean']
                        adjusted_return_tstats[k1,k2]=summary['tstat']
                    else:
                        adjusted_returns[:,k1, k2] = excess_return[:, 0] - dot(factors,ols_results['betas'].reshape(-1,1))[:,0]
                        summary=nantstat(adjusted_returns)
                        adjusted_return_mean[k1,k2]=summary['mean']
                        adjusted_return_tstats[k1, k2] = summary['tstat']
                except Exception as e:
                    print(e)
    elif len(dims)==3:
        for k1 in range(0,K1):
            for k2 in range(0,K2):
                for k3 in range(0,K3):
                    try:
                        if k2!=K2-1 and k3!=K3-1:
                            excess_return=returns[:,k1,k2,k3].reshape(-1, 1)-rf
                        else:
                            excess_return=returns[:,k1,k2,k3].reshape(-1, 1)
                        ols_results=nanols(excess_return,factors,constant)
                        if constant==True:
                            adjusted_returns[:,k1,k2,k3]=(ols_results['resids']+ols_results['betas'][0])[:,0]
                            summary = nantstat(adjusted_returns[:, k1, k2,k3])
                            adjusted_return_mean[k1, k2, k3] = summary['mean']
                            adjusted_return_tstats[k1, k2, k3] = summary['tstat']
                        else:
                            adjusted_returns[:,k1, k2,k3] = ols_results['resids'][:,0]
                            summary=nantstat(adjusted_returns)
                            adjusted_return_mean[k1, k2, k3]=summary['mean']
                            adjusted_return_tstats[k1, k2, k3] = summary['tstat']
                    except Exception as e:
                        print(e)
    return {'alpha':adjusted_return_mean,'tstat':adjusted_return_tstats,'adj_returns':adjusted_returns}
def get_risk_adjusted_stock_returns(returns,factors,constant=True,robust_type='HC0',factorName='',writePickle=False):
    '''This function adjusts the individual security returns by risk factors. It's not recommended as individual security returns might be more noisier than portfolio returns. If number of
    observations is less than 24, then ignore that security'''
    T, N = returns.shape
    adjusted_returns=full([T,N],nan)
    rf=factors[:,0:1]
    factors=factors[:,1:]
    for i in range(0, N):
        try:
            excess_return=returns[:,i].reshape(-1, 1)-rf
            ols_results=nanols(excess_return,factors,constant)
            if ols_results['nobs']>=24:
                if constant==True:
                    adjusted_returns[:,i]=(ols_results['resids']+ols_results['betas'][0])[:,0]
                else:
                    adjusted_returns[:,i] =ols_results['resids'][:,0]
        except:
            pass
    if writePickle:
        write_pickle(matricesPath+factorName+'_ADJ_TOTRET.txt',adjusted_returns)
    return adjusted_returns
def set_risk_adjusted_stock_returns(struct,factorsList,rawReturnField='TOTRET',constant=True,robust_type='HC0',outputType='abc'):
    raw=struct[rawReturnField]
    for factorName in factorsList:
        factor=struct[factorName]
        adj_returns=get_risk_adjusted_stock_returns(raw,factor)
        if outputType=='numpy':
            write_pickle(matricesPath + factorName+ '_ADJ_TOTRET.txt', adj_returns)
            return 0
        else:
            struct[factorName+'_ADJ_TOTRET']=adj_returns
    return struct
def get_factors(dateStart,dateEnd,factors,dateField='YYYYMM',types =['CAPM','FF3','FF5','CARHART','ZB-STRAD-INDEX']):
    factors.columns=[col.upper() for col in factors.columns]
    factors=factors[ (factors[dateField]>=dateStart) & (factors[dateField]<=dateEnd)]
    factors_dict=dict()
    factors_dict['DATE']=np.sort(factors['YYYYMM'].unique())
    for type in types:
        if type=='CAPM':
            factorColumns=['RF','MKTRF']
        elif type=='FF3':
            factorColumns=['RF','MKTRF','SMB','HML']
        elif type=='CARHART':
            factorColumns = ['RF','MKTRF', 'SMB', 'HML', 'UMD']
        elif type=='FF5':
            factorColumns=['RF','MKTRF', 'SMB', 'HML', 'RMW','CMA']
        elif type=='SYY':
            factorColumns=['RF','MKTRF','SMB','MGMT','PERF']
        elif type=='Q':
            factorColumns=['RF','MKT','ME','IA','ROE']
        elif type=='LIQ':
            factorColumns=['RF','SMB','HML','PS_INNOV']
        elif type=='ZB-STRAD-INDEX':
            factorColumns=['RF','ZB-STRAD-INDEX']
        a=factors[factorColumns].values
        factors_dict[type]=a

    return factors_dict
if __name__ == "__main__":
    pass