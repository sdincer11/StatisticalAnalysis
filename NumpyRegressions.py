
import statsmodels.api as sm
from numpy import *
def ols(y,x,addConstant=True,robust='nonrobust',cov_keywords=None):
    if addConstant==True:
        x=sm.add_constant(x)
    model=sm.OLS(y,x,missing='drop')
    results=model.fit(cov_type=robust,cov_kwds=cov_keywords)
    return results
def nanols(y,x,constant=True,fixed_effects=None,robustness='HC0',cov_keywords=None):
    T, N=y.shape
    _dummy,no_indep_vars=x.shape
    results=dict()
    if fixed_effects is not None:
        T_fixed, N_fixed = fixed_effects.shape
        x_all=x
        for i in range (0,N_fixed):
            not_nan_idx=where(~isnan(fixed_effects[:,i]))[0]
            dummy=sm.categorical(fixed_effects[not_nan_idx,i],drop=True)
            _,dummy_N=dummy.shape
            fe=full([T_fixed,dummy_N],nan)
            fe[not_nan_idx,:]=dummy
            x_all=hstack((x_all,fe[:,1:]))
    else:
        x_all=x
    stack = hstack((y, x_all))
    non_miss_idx = all(~isnan(stack), axis=1).reshape(-1, 1)
    results['resids'] = full([T, 1], nan)
    results['yhats']=full([T,1],nan)
    results['rsquared']=nan
    results['rsquared_adj']=nan
    results['nobs']=nan
    results['non_miss_index']=nan
    if constant==True:
        no_indep_vars=no_indep_vars+1
    results['betas']=full([1,no_indep_vars],nan)
    results['tstats']=full([1,no_indep_vars],nan)
    results['pvals']=full([1,no_indep_vars],nan)


    if any(non_miss_idx):
        ols_results=ols(y,x_all,constant,robustness,cov_keywords)
        a=where(non_miss_idx)[0]
        results['resids'][a,:]=ols_results.resid.reshape(-1,1)
        results['tstats']=ols_results.tvalues[0:no_indep_vars]
        results['rsquared']=ols_results.rsquared
        results['rsquared_adj']=ols_results.rsquared_adj
        results['nobs']=int(ols_results.nobs)
        results['pvals']=ols_results.pvalues[0:no_indep_vars]
        results['non_miss_index']=where(non_miss_idx)[0]
        results['betas']=ols_results.params[0:no_indep_vars]
        results['yhats']=ols_results.fittedvalues
    return results
def nantstat(timeSeries,axisVal=0):
    'It takes a numpy array as input, and returns its mean, std and tstat as output'
    result=dict()
    result['mean']=nanmean(timeSeries,axis=axisVal)
    result['std']=nanstd(timeSeries,axis=axisVal)
    N=count_nonzero(~isnan(timeSeries),axis=axisVal)
    result['tstat']=result['mean']/result['std']*sqrt(N)
    result['nobs']=N
    return result

if __name__ == "__main__":
    pass