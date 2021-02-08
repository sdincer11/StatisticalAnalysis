import os
os.system('python -m venv venv && venv\\Scripts\\activate.bat && pip install pipreqs && pipreqs "' + os.getcwd() +'" && pip install -r requirements.txt')
from numpy import *
import numpy as np
from LatexGenerator import *
from FactorAdjustments import *
from NumpyRegressions import *
import pandas as pd
class PortfolioSort():
    def __init__(self, csvFile, factorsFile, variablesList, factorsList = ['CAPM','FF3'], tstart=None, tend=None, returnField='TOTRET', dateField='YYYYMM', idField='PERMNO', portfolioWeightField='SIZE'):
        try:
            self.dF = pd.read_csv(csvFile)
        except:
            print('Please make sure the input CSV file for firm data exists in the specified location')
            return
        self.dateField = dateField
        self.idField = idField
        self.tstart = self.dF[self.dateField].min() if tstart == None else tstart
        self.tend = self.dF[self.dateField].max() if tend == None else tend
        self.dF = self.dF[self.dF[dateField].between(tstart, tend)]
        try:
            self.factorsDF = pd.read_csv(factorsFile)
            self.factors = get_factors(self.tstart, self.tend, self.factorsDF, self.dateField, types=factorsList)
            self.factors = {key:val for key,val in self.factors.items() if key!='DATE'}
        except Exception as e:
            print(e)
            return
        self.portfolioWeightField = portfolioWeightField
        self.variables = variablesList + [returnField] if self.portfolioWeightField == None else variablesList + [returnField, portfolioWeightField]
        self.matrices = self.create_matrices_from_dataframe()
        self.projectPath = os.getcwd().replace('\\','/')+'/'
        self.latexPath=self.projectPath+'latex/'
        self.latexTablesPath=self.latexPath+'tables/'
        self.latexFiguresPath=self.latexPath+'figures/'
        if not os.path.exists(self.latexPath):
            os.mkdir(self.latexPath)
        if not os.path.exists(self.latexTablesPath):
            os.mkdir(self.latexTablesPath)
        if not os.path.exists(self.latexFiguresPath):
            os.mkdir(self.latexFiguresPath)

    def create_matrices_from_dataframe(self):
        matrices = dict()
        try:
            primary_keys = [ self.dateField, self.idField]
            firmIdDate = self.dF[primary_keys]
            firmIdDate = firmIdDate[(firmIdDate[self.dateField] >= self.tstart) & (firmIdDate[self.dateField] <= self.tend)]
            dates = pd.Series(firmIdDate[self.dateField].drop_duplicates()).sort_values().reset_index(drop=True).values
            ids = pd.Series(firmIdDate[self.idField].drop_duplicates()).sort_values().reset_index(drop=True).values
            matrices[self.idField] = ids
            matrices[self.dateField] = dates
            data = self.dF[(self.dF[self.dateField] >= self.tstart) & (self.dF[self.dateField] <= self.tend)]
            data.sort_values(by=[self.idField, self.dateField], inplace=True)
            try:
                for value in self.variables:
                    data_matrix = array(data[[self.idField, self.dateField, value]].drop_duplicates().pivot(index=self.dateField, columns=self.idField,values=value).values)
                    if data_matrix.dtype == 'object':
                        data_matrix.astype('float')
                    matrices[value] = data_matrix
            except Exception as E:
                print(E)
        except Exception as e:
            print(e)
        return matrices
    def output_double_sort(self, returns, K1, K2, sortParam1, sortParam2, label_y_axis, label_x_axis, factors, weights= None, lag=0, precision=2, addTstat=True, firm_filter=None, time_filter=None, tableNameSuffix='', sort_type='independent', outputStars=False, risk_adjustment=True, displayPercent=True):
        resultDict=dict()
        header = [label_y_axis + str(i) for i in range(1, K1 + 1)] + [label_y_axis + str(K1) + ' - ' + label_y_axis + '1']
        if time_filter is not None:
            factors2={key:val[time_filter] for key,val in factors.items()}
        else:
            factors2={key:val for key,val in factors.items()}
        if sort_type.upper()=='INDEPENDENT':
            sortedPortfolios = self.double_sort(returns, K1, K2, sortParam1, sortParam2, weights, lag, bool_filter_indices=firm_filter, time_indices=time_filter)
        elif sort_type.upper()=='SEQUENTIAL':
            sortedPortfolios=self.double_sort_sequential(returns, K1, K2, sortParam1, sortParam2, weights, lag, bool_filter_indices=firm_filter, time_indices=time_filter)
            tableNameSuffix+='_SEQUENTIAL'

        resultDict['EW_RAW_RETURNS']=sortedPortfolios['ew_returns']
        resultDict['EW_RAW_MEANS']=sortedPortfolios['ew_raw_mean']
        resultDict['EW_RAW_TSTATS'] = sortedPortfolios['ew_raw_tstat']
        if displayPercent:
            resultDict['EW_RAW_MEANS'] *= 100
        if not addTstat:
            resultDict['EW_RAW_TSTATS'] = array([])
        table_coef_tstatfixed(resultDict['EW_RAW_MEANS'], resultDict['EW_RAW_TSTATS'], header, precision,self.latexTablesPath + 'EW' + '_RAW_' + label_y_axis + '_by_' + label_x_axis + '_' + str(K1) + '_' + str(K2) + tableNameSuffix, addStars=outputStars)

        if weights !=None:
            resultDict['VW_RAW_RETURNS'] = sortedPortfolios['vw_returns']
            resultDict['VW_RAW_MEANS'] = sortedPortfolios['vw_raw_mean']
            resultDict['VW_RAW_TSTATS'] = sortedPortfolios['vw_raw_tstat']
            if displayPercent:
                resultDict['VW_RAW_MEANS'] *= 100
            if not addTstat:
                resultDict['VW_RAW_TSTATS'] = array([])
            table_coef_tstatfixed(resultDict['VW_RAW_MEANS'], resultDict['VW_RAW_TSTATS'], header, precision,self.latexTablesPath + 'VW' + '_RAW_' + label_y_axis + '_by_' + label_x_axis + '_' + str(K1) + '_' + str(K2) + tableNameSuffix, addStars=outputStars)


        if risk_adjustment==True:
            for factor_name,factor in factors2.items():
                ew_adjusted_portfolios=get_risk_adjusted_portfolio_returns(sortedPortfolios['ew_returns'],factor)
                resultDict['EW_'+factor_name+'_RETURNS']=ew_adjusted_portfolios['adj_returns']
                resultDict['EW_'+factor_name+'_MEANS'] = ew_adjusted_portfolios['alpha']
                resultDict['EW_' + factor_name + '_TSTATS']=ew_adjusted_portfolios['tstat']
                if displayPercent:
                    resultDict['EW_'+factor_name+'_MEANS']=resultDict['EW_'+factor_name+'_MEANS']*100
                if not addTstat:
                    resultDict['EW_RAW_TSTATS'] = array([])
                table_coef_tstatfixed(resultDict['EW_' + factor_name + '_MEANS'],resultDict['EW_' + factor_name + '_TSTATS'], header, precision,self.latexTablesPath + 'EW_' + factor_name + '_' + label_y_axis + '_by_' + label_x_axis + '_' + str(K1) + '_' + str(K2) + tableNameSuffix, addStars=outputStars)
                if weights != None:
                    vw_adjusted_portfolios = get_risk_adjusted_portfolio_returns(sortedPortfolios['vw_returns'], factor)
                    resultDict['VW_' + factor_name + '_RETURNS'] = vw_adjusted_portfolios['adj_returns']
                    resultDict['VW_' + factor_name + '_MEANS'] = vw_adjusted_portfolios['alpha']
                    resultDict['VW_' + factor_name + '_TSTATS'] = vw_adjusted_portfolios['tstat']
                    if displayPercent:
                        resultDict['VW_' + factor_name + '_MEANS'] *= 100
                    if not addTstat:
                        resultDict['VW_'+ factor_name + '_TSTATS'] = array([])
                    table_coef_tstatfixed(resultDict['VW_' + factor_name + '_MEANS'],resultDict['VW_' + factor_name + '_TSTATS'], header, precision,self.latexTablesPath + 'VW_' + factor_name + '_' + label_y_axis + '_by_' + label_x_axis + '_' + str( K1) + '_' + str(K2) + tableNameSuffix, addStars=outputStars)

        resultDict['NOBS_AVG']=sortedPortfolios['nobs_avg']
        resultDict['IDX']=sortedPortfolios['idx']
        table_coef_tstatfixed(resultDict['NOBS_AVG'],array([]),header,precision,self.latexTablesPath+'NOBS_AVG'+'_'+label_y_axis+'_by_'+label_x_axis+'_'+str(K1)+'_'+str(K2) +tableNameSuffix,addStars=outputStars)
        return resultDict
    def output_triple_sort(self, returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, label1, label2, label3, factors, weights=None, lag=0, precision=2, addTstat=True, firm_filter=None,time_filter=None, tableNameSuffix='', sort_type='independent',outputStars=False, risk_adjustment=True, displayPercent=True):
        resultDict=dict()
        header = [label2 + str(i) for i in range(1, K2 + 1)] + [label2 + str(K2) + ' - ' + label2 + '1']
        if time_filter is not None:
            factors2={key:val[time_filter] for key,val in factors.items()}
        else:
            factors2={key:val for key,val in factors.items()}
        if sort_type=='independent':
            sortedPortfolios = self.triple_sort(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights,lag,bool_filter_indices=firm_filter, time_indices=time_filter)
        elif sort_type=='sequential':
            sortedPortfolios = self.triple_sort_sequential(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights, lag,bool_filter_indices=firm_filter, time_indices=time_filter)
            tableNameSuffix+='_SEQUENTIAL'
        elif sort_type=='sequential_independent':
            sortedPortfolios = self.triple_sort_sequential_independent(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights, lag,bool_filter_indices=firm_filter, time_indices=time_filter)
            tableNameSuffix+='_SEQUENTIAL_INDEPENDENT'

        resultDict['EW_RAW_RETURNS'] = sortedPortfolios['ew_returns']
        resultDict['EW_RAW_MEANS'] = sortedPortfolios['ew_raw_mean']
        resultDict['EW_RAW_TSTATS'] = sortedPortfolios['ew_raw_tstat']
        if displayPercent:
            resultDict['EW_RAW_MEANS'] *= 100
        if not addTstat:
            resultDict['EW_RAW_TSTATS'] = array([])
        table_coef_tstatfixed(np.hstack((resultDict['EW_RAW_MEANS'][0,:,:],resultDict['EW_RAW_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['EW_RAW_TSTATS'][0,:,:],resultDict['EW_RAW_TSTATS'][K1-1,:,:])), header, precision,self.latexTablesPath + 'EW' + '_RAW_' + label1 + '_by_' + label2 +'_by_'+label3+'_' + str(K1) + '_' + str(K2)+'_'+str(K3)+tableNameSuffix,addStars=outputStars)

        if weights != None:
            resultDict['VW_RAW_RETURNS'] = sortedPortfolios['vw_returns']
            resultDict['VW_RAW_MEANS'] = sortedPortfolios['vw_raw_mean']
            resultDict['VW_RAW_TSTATS'] = sortedPortfolios['vw_raw_tstat']
            if displayPercent:
                resultDict['VW_RAW_MEANS'] *= 100
            if not addTstat:
                resultDict['VW_RAW_TSTATS'] = array([])
            table_coef_tstatfixed(np.hstack((resultDict['VW_RAW_MEANS'][0,:,:],resultDict['VW_RAW_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['VW_RAW_TSTATS'][0,:,:],resultDict['VW_RAW_TSTATS'][K1-1,:,:])), header, precision,self.latexTablesPath + 'VW' + '_RAW_' + label1 + '_by_' + label2 +'_by_'+label3+'_' + str(K1) + '_' + str(K2)+'_'+str(K3)+tableNameSuffix,addStars=outputStars)

        if risk_adjustment:
            for factor_name,factor in factors2.items():
                ew_adjusted_portfolios=get_risk_adjusted_portfolio_returns(sortedPortfolios['ew_returns'],factor)
                vw_adjusted_portfolios = get_risk_adjusted_portfolio_returns(sortedPortfolios['vw_returns'],factor)
                resultDict['EW_' + factor_name +'_RETURNS']=ew_adjusted_portfolios['adj_returns']
                resultDict['EW_' + factor_name + '_MEANS'] = ew_adjusted_portfolios['alpha']
                resultDict['EW_' + factor_name + '_TSTATS']=ew_adjusted_portfolios['tstat']
                if displayPercent:
                    resultDict['EW_'+ factor_name + '_MEANS'] *= 100
                if not addTstat:
                    resultDict['EW_' + factor_name +'_TSTATS'] = array([])
                table_coef_tstatfixed(np.hstack((resultDict['EW_' + factor_name + '_MEANS'][0, :, :],resultDict['EW_' + factor_name + '_MEANS'][K1 - 1, :, :])) * 100, np.hstack((resultDict['EW_' + factor_name + '_TSTATS'][0, :, :], resultDict['EW_' + factor_name + '_TSTATS'][K1 - 1, :, :])), header, precision, self.latexTablesPath + 'EW_' + factor_name + '_' + label1 + '_by_' + label2 + '_by_' + label3 + '_' + str( K1) + '_' + str(K2) + '_' + str(K3) + tableNameSuffix, addStars=outputStars)

                if weights != None:
                    resultDict['VW_' + factor_name + '_RETURNS'] = vw_adjusted_portfolios['adj_returns']
                    resultDict['VW_' + factor_name + '_MEANS'] = vw_adjusted_portfolios['alpha']
                    resultDict['VW_' + factor_name + '_TSTATS'] = vw_adjusted_portfolios['tstat']
                    if displayPercent:
                        resultDict['VW_' + factor_name + '_MEANS'] *= 100
                    if not addTstat:
                        resultDict['VW_' + factor_name + '_TSTATS'] = array([])
                    table_coef_tstatfixed(np.hstack((resultDict['VW_'+factor_name+'_MEANS'][0,:,:],resultDict['VW_'+factor_name+'_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['VW_' + factor_name + '_TSTATS'][0,:,:],resultDict['VW_' + factor_name + '_TSTATS'][K1-1,:,:])), header, precision,self.latexTablesPath + 'VW_' +factor_name+'_'+ label1 + '_by_' + label2 +'_by_' + label3 + '_' + str(K1) + '_' + str(K2) + '_' + str(K3) +tableNameSuffix,addStars=outputStars)

        resultDict['NOBS_AVG'] = sortedPortfolios['nobs_avg']
        resultDict['IDX']=sortedPortfolios['idx']
        table_coef_tstatfixed(np.hstack((resultDict['NOBS_AVG'] [0,:,:],resultDict['NOBS_AVG'][K1-1,:,:])), array([]), header, precision,self.latexTablesPath + 'NOBS_AVG' + '_' + label1 + '_by_' + label2 +'_by_' + label3 + '_' + str( K1) + '_' + str(K2) + '_' + str(K3)+tableNameSuffix,addStars=outputStars)
        return resultDict
    def output_single_sort(self, returns, K, sortParam, label_y_axis, factors, weights = None, lag=0, precision=2, addTstat=True, firm_filter=None, time_filter=None, tableNameSuffix='', outputStars=False, risk_adjustment=True, displayStat='mean', displayPercent=True):
        resultDict=dict()
        header = [label_y_axis + str(i) for i in range(1, K + 1)] + [label_y_axis + str(K) + ' - ' + label_y_axis + '1']
        if time_filter is not None:
            factors2={key:val[time_filter] for key,val in factors.items()}
        else:
            factors2={key:val for key,val in factors.items()}
        sortedPortfolios = self.single_sort(returns, K, sortParam, weights, lag, firm_filter_indices=firm_filter, time_indices=time_filter,display=displayStat)
        resultDict['EW_RAW_RETURNS']=sortedPortfolios['ew_returns']
        resultDict['EW_RAW_MEANS']=sortedPortfolios['ew_raw_mean']
        resultDict['EW_RAW_TSTATS']=sortedPortfolios['ew_raw_tstat']
        if not addTstat:
            resultDict['EW_RAW_TSTATS'] = array([])
        if displayPercent:
            resultDict['EW_RAW_MEANS']*= 100
        table_coef_tstatfixed(resultDict['EW_RAW_MEANS'], resultDict['EW_RAW_TSTATS'], header, precision,self.latexTablesPath + 'EW' + '_RAW_' + label_y_axis + '_by_' + str(K) + tableNameSuffix, addStars=outputStars)

        if weights != None:
            resultDict['VW_RAW_RETURNS'] = sortedPortfolios['vw_returns']
            resultDict['VW_RAW_MEANS'] = sortedPortfolios['vw_raw_mean']
            resultDict['VW_RAW_TSTATS'] = sortedPortfolios['vw_raw_tstat']
            if not addTstat:
                resultDict['VW_RAW_TSTATS'] = array([])
            if displayPercent:
                resultDict['VW_RAW_MEANS'] *= 100
            table_coef_tstatfixed(resultDict['VW_RAW_MEANS'], resultDict['VW_RAW_TSTATS'], header, precision,self.latexTablesPath + 'VW'+'_RAW_'+label_y_axis+'_by_'+str(K)+tableNameSuffix,addStars=outputStars)

        if risk_adjustment==True:
            for factor_name,factor in factors2.items():
                ew_adjusted_portfolios = get_risk_adjusted_portfolio_returns(sortedPortfolios['ew_returns'],factor)
                vw_adjusted_portfolios = get_risk_adjusted_portfolio_returns(sortedPortfolios['vw_returns'],factor)
                resultDict['EW_' + factor_name + '_RETURNS'] = ew_adjusted_portfolios['adj_returns']
                resultDict['EW_' + factor_name + '_MEANS'] = ew_adjusted_portfolios['alpha']
                resultDict['EW_' + factor_name + '_TSTATS'] = ew_adjusted_portfolios['tstat']
                if not addTstat:
                    resultDict['EW_RAW_TSTATS'] = array([])
                if displayPercent:
                    resultDict['EW_' + factor_name + '_MEANS'] *= 100
                table_coef_tstatfixed(resultDict['EW_' + factor_name + '_MEANS'], resultDict['EW_' + factor_name + '_TSTATS'], header, precision,self.latexTablesPath + 'EW_' + factor_name + '_' + label_y_axis + '_by_' + str(K) + tableNameSuffix, addStars=outputStars)
                if weights != None:
                    resultDict['VW_' + factor_name + '_RETURNS'] = vw_adjusted_portfolios['adj_returns']
                    resultDict['VW_' + factor_name + '_MEANS'] = vw_adjusted_portfolios['alpha']
                    resultDict['VW_' + factor_name + '_TSTATS'] = vw_adjusted_portfolios['tstat']
                    if not addTstat:
                        resultDict['VW_'+ factor_name +'_TSTATS'] = array([])
                    if displayPercent:
                        resultDict['VW_' + factor_name + '_MEANS'] *= 100
                    table_coef_tstatfixed(resultDict['VW_'+factor_name+'_MEANS'], resultDict['VW_'+factor_name+'_TSTATS'], header, precision,self.latexTablesPath + 'VW_' + factor_name + '_' + label_y_axis+'_by_'+str(K) + tableNameSuffix,addStars=outputStars)
        resultDict['IDX']=sortedPortfolios['idx']
        resultDict['NOBS_AVG']=sortedPortfolios['nobs_avg']
        table_coef_tstatfixed(resultDict['NOBS_AVG'],array([]),header,precision,self.latexTablesPath+'NOBS_AVG'+'_'+label_y_axis+'_by_'+str(K) +tableNameSuffix)
        return resultDict
    def single_sort(self, returns, K, sort_param, weights=None, lag=0,firm_filter_indices=None, time_indices=None, display='mean'):
        results=dict()
        T, N = sort_param.shape
        nobs = full([T, K,1], nan)
        nobs_avg = full([K,1], nan)

        portfolio_return_EW = full([T, K + 1,1], nan)
        portfolio_return_VW = full([T, K + 1,1], nan)

        EW_raw_mean = full([K + 1,1], nan)
        EW_raw_tstat = full([K + 1,1], nan)

        if weights != None:
            VW_raw_mean = full([K + 1,1], nan)
            VW_raw_tstat = full([K + 1,1], nan)


        bounds = full([T, K + 1], nan)
        idx=[]
        for t in range(1, T):
            q1 = nanpercentile(sort_param[t - 1, :], linspace(0, 100, K + 1))

            for i, el in enumerate(q1[:-1]):
                if q1[i] >= q1[i + 1]:
                    q1[i + 1] = q1[i] + 1
            bounds[t - 1, :] = q1
            for k1 in range(0, K):
                if k1 < K - 1:
                    a = where((sort_param[t - 1, :] >= q1[k1]) & (sort_param[t - 1, :] < q1[k1 + 1]))[0]

                elif k1 == K - 1:
                    a = where((sort_param[t - 1, :] >= q1[k1]) & (sort_param[t - 1, :] <= q1[k1 + 1]))[0]
                if firm_filter_indices is not None:
                    b = where(firm_filter_indices[t - 1, :] == True)[0]
                    temp = a[in1d(a, b)]
                else:
                    temp = a
                temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                idx.append({'T':t-1,'K':k1,'PERMNO_IDX':temp})
                nobs[t - 1, k1,0] = len(temp)
                if display=='mean':
                    portfolio_return_EW[t-1, k1,0] = nanmean(returns[t - lag, temp])
                elif display=='median':
                    portfolio_return_EW[t-1, k1,0] = nanmedian(returns[t - lag, temp])
                if weights != None:
                    portfolio_return_VW[t-1, k1,0] =nansum(returns[t-lag,temp]*weights[t-1,temp])/nansum(weights[t - 1, temp])

                '''The following two lines are not used always, but meaningful only when you do characteristic adjustment
                portfolio_adj_return_EW[t,a]=returns[t,a]-portfolio_return_EW[t,k1,k2]
                portfolio_adj_return_VW[t,a]=returns[t,a]-portfolio_return_VW[t,k1,k2]'''

            portfolio_return_EW[t-1, K,0] = portfolio_return_EW[t-1, K - 1,0] - portfolio_return_EW[t-1,0,0]
            if weights != None:
                portfolio_return_VW[t-1, K,0] = portfolio_return_VW[t-1, K - 1,0] - portfolio_return_VW[t-1,0,0]

        if time_indices is not None:
            portfolio_return_EW = portfolio_return_EW[time_indices,:,:]
            if weights != None:
                portfolio_return_VW = portfolio_return_VW[time_indices,:,:]
            nobs=nobs[time_indices,:,:]
            bounds=bounds[time_indices,:]

        for k1 in range(0, K + 1):
            if k1 < K:
                nobs_avg[k1,0] = nantstat(nobs[:,k1,0])['mean']
            summary_EW = nantstat(portfolio_return_EW[:,k1,0])
            EW_raw_mean[k1,0] = summary_EW['mean']
            EW_raw_tstat[k1,0] = summary_EW['tstat']

            if weights != None:
                summary_VW = nantstat(portfolio_return_VW[:, k1,0])
                VW_raw_mean[k1,0] = summary_VW['mean']
                VW_raw_tstat[k1,0] = summary_VW['tstat']

            '''The following lines are to adjust value weighted and equal weighted portfolios for risk factors'''

        results['bounds'] = bounds

        results['ew_returns'] = portfolio_return_EW
        results['ew_raw_mean'] = EW_raw_mean
        results['ew_raw_tstat'] = EW_raw_tstat

        if weights != None:
            results['vw_returns'] = portfolio_return_VW
            results['vw_raw_mean'] = VW_raw_mean
            results['vw_raw_tstat'] = VW_raw_tstat

        results['idx']=idx
        results['nobs'] = nobs
        results['nobs_avg'] = nobs_avg

        return results
    def double_sort(self, returns, K1, K2, sort_param1, sort_param2, weights = None, lag=0, bool_filter_indices=None, time_indices=None):
        '''
        % function r=get_adjusted_returns_two(crsp,K1,K2,adjustment_factor1,adjustment_factor2,tstart)
        THIS FUNCTION ADJUSTS BY TWO INDEPENDENT FACTORS
         crsp      = structure with return and mv data
        K         = number of size groups
        % adjustment_factor = category by which we are adjusting
        '''
        results=dict()
        T,N=sort_param1.shape

        nobs=full([T,K1,K2],nan)
        nobs_avg=full([K1,K2],nan)

        portfolio_return_EW=full([T,K1+1,K2+1],nan)
        portfolio_return_VW=full([T,K1+1,K2+1],nan)

        EW_raw_mean = full([K1 + 1, K2 + 1], nan)
        EW_raw_tstat = full([K1 + 1, K2 + 1], nan)
        
        if weights!=None:
            VW_raw_mean = full([K1 + 1, K2 + 1], nan)
            VW_raw_tstat = full([K1 + 1, K2 + 1], nan)

        bounds1=full([T,K1+1],1)
        bounds2=full([T,K2+1],1)

        idx=[]
        for t in range(1,T):
            a=linspace(0,100,K1+1)
            q1=nanpercentile(sort_param1[t-1,:],linspace(0,100,K1+1))
            tt=nanmin(sort_param1[t-1,:])
            for i,el in enumerate(q1[:-1]):
                if q1[i]>=q1[i+1]:
                    q1[i+1]=q1[i + 1] + 1
            q2=nanpercentile(sort_param2[t-1,:],linspace(0,100,K2+1))
            for i,el in enumerate(q2[:-1]):
                if q2[i]>=q2[i+1]:
                    q2[i+1]=q2[i + 1] + 1
            bounds1[t-1,:]=q1
            bounds2[t-1,:]=q2
            for k1 in range(0,K1):
                for k2 in range(0,K2):
                    if k1<K1-1:
                        if k2<K2-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<q1[k1+1]) & (sort_param2[t-1,:]>=q2[k2]) & (sort_param2[t-1,:]<q2[k2+1]) & (~isnan(returns[t-lag,:])) )[0]
                        elif k2==K2-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<q1[k1+1]) & (sort_param2[t-1,:]>=q2[k2]) & (sort_param2[t-1,:]<=q2[k2+1])& (~isnan(returns[t-lag,:])) )[0]

                    elif k1==K1-1:
                        if k2<K2-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<=q1[k1+1]) & (sort_param2[t-1,:]>=q2[k2]) & (sort_param2[t-1,:]<q2[k2+1]) & (~isnan(returns[t-lag,:])) )[0]
                        elif k2==K2-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<=q1[k1+1]) & (sort_param2[t-1,:]>=q2[k2]) & (sort_param2[t-1,:]<=q2[k2+1]) & (~isnan(returns[t-lag,:])) )[0]
                    if bool_filter_indices is not None:
                        b = where(bool_filter_indices[t - 1, :] == True)[0]
                        temp = a[in1d(a, b)]
                    else:
                        temp=a
                    if len(temp)>0:
                        k=2
                    temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                    idx.append({'T':t-1,'K1':k1,'K2':k2,'PERMNO_IDX':temp})
                    nobs[t-1,k1,k2]=len(temp)
                    portfolio_return_EW[t-1,k1,k2]=nanmean(returns[t-lag,temp])
                    if weights != None:
                        portfolio_return_VW[t-1, k1, k2]=nansum(returns[t-lag,temp]*weights[t-1,temp])/nansum(weights[t-1,temp])
            for k2 in range(0, K2+1):
                if K1 > 1:
                    portfolio_return_EW[t-1, K1, k2] = portfolio_return_EW[t-1, K1 - 1, k2] - portfolio_return_EW[t-1, 0, k2]
                    if weights != None:
                        portfolio_return_VW[t-1, K1, k2] = portfolio_return_VW[t-1, K1 - 1, k2] - portfolio_return_VW[t-1, 0, k2]
            for k1 in range(0, K1+1):
                if K2 > 1:
                    portfolio_return_EW[t-1, k1, K2] = portfolio_return_EW[t-1, k1, K2 - 1] - portfolio_return_EW[t-1, k1, 0]
                    if weights != None:
                        portfolio_return_VW[t-1, k1, K2] = portfolio_return_VW[t-1, k1, K2 - 1] - portfolio_return_VW[t-1, k1, 0]
        if time_indices is not None:
            portfolio_return_EW = portfolio_return_EW[time_indices,:,:]

            if weights != None:
                portfolio_return_VW = portfolio_return_VW[time_indices,:,:]
            nobs=nobs[time_indices,:,:]
            bounds1=bounds1[time_indices,:]
            bounds2=bounds2[time_indices,:]
            idx=[element for element in idx if element['T'] in time_indices]

        for k1 in range(0, K1 + 1):
            for k2 in range(0, K2 + 1):
                if k1<K1 and k2<K2:
                    nobs_avg[k1, k2] = nantstat(nobs[:, k1, k2])['mean']
                summary_EW=nantstat(portfolio_return_EW[:, k1, k2])
                EW_raw_mean[k1, k2] = summary_EW['mean']
                EW_raw_tstat[k1, k2] = summary_EW['tstat']

                if weights != None:
                    summary_VW=nantstat(portfolio_return_VW[:, k1, k2])
                    VW_raw_mean[k1, k2] = summary_VW['mean']
                    VW_raw_tstat[k1, k2] = summary_VW['tstat']


        '''The following lines are to adjust value weighted and equal weighted portfolios for risk factors'''


        results['bounds1'] = bounds1
        results['bounds2'] = bounds2
        results['idx']=idx
        results['ew_returns'] = portfolio_return_EW
        results['ew_raw_mean']=EW_raw_mean
        results['ew_raw_tstat']=EW_raw_tstat
        results['nobs'] = nobs
        results['nobs_avg']=nobs_avg
        if weights != None:
            results['vw_returns'] = portfolio_return_VW
            results['vw_raw_mean'] = VW_raw_mean
            results['vw_raw_tstat'] = VW_raw_tstat

        return results
    def double_sort_sequential(self, returns, K1, K2, sort_param1, sort_param2, weights=None, lag=0, bool_filter_indices=None, time_indices=None):
        '''
    % function r=get_adjusted_returns_two(crsp,K1,K2,adjustment_factor1,adjustment_factor2,tstart)
    THIS FUNCTION ADJUSTS BY TWO INDEPENDENT FACTORS
     crsp      = structure with return and mv data
    K         = number of size groups
    % adjustment_factor = category by which we are adjusting
    '''
        results=dict()
        T,N=sort_param1.shape

        nobs=full([T,K1,K2],nan)
        nobs_avg=full([K1,K2],nan)

        portfolio_return_EW=full([T,K1+1,K2+1],nan)
        portfolio_return_VW=full([T,K1+1,K2+1],nan)

        EW_raw_mean = full([K1 + 1, K2 + 1], nan)
        EW_raw_tstat = full([K1 + 1, K2 + 1], nan)

        if weights != None:
            VW_raw_mean = full([K1 + 1, K2 + 1], nan)
            VW_raw_tstat = full([K1 + 1, K2 + 1], nan)

        bounds1=full([T,K1+1],1)
        bounds2=full([T,K2+1],1)

        idx=[]
        try:
            for t in range(1,T):
                if t==186 or t==185:
                    sinan=5
                a=linspace(0,100,K1+1)
                q1=nanpercentile(sort_param1[t-1,:],linspace(0,100,K1+1))
                for i,el in enumerate(q1[:-1]):
                    if q1[i]>=q1[i+1]:
                        q1[i+1]=q1[i] + 1
                bounds1[t-1,:]=q1

                for k1 in range(0,K1):
                        if k1<K1-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<q1[k1+1]))[0]

                        elif k1==K1-1:
                            a=where((sort_param1[t-1,:]>=q1[k1]) & (sort_param1[t-1,:]<=q1[k1+1]))[0]
                        if len(a)>0:
                            q2 = nanpercentile(sort_param2[t - 1, a], linspace(0, 100, K2 + 1))
                            for i, el in enumerate(q2[:-1]):
                                if q2[i] >= q2[i + 1]:
                                    q2[i + 1] = q2[i + 1] + 1
                            bounds2[t - 1, :] = q2
                            for k2 in range(0, K2):
                                if k2 < K2 - 1:
                                    b = a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] < q2[k2 + 1]))[0]]
                                elif k2==K2-1:
                                    b = a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] <= q2[k2 + 1]))[0]]
                                if bool_filter_indices is not None:
                                    c = where(bool_filter_indices[t - 1, :] == True)[0]
                                    temp = b[in1d(b, c)]
                                else:
                                    temp=b
                                temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                                idx.append({'T':t-1,'K1':k1,'K2':k2,'PERMNO_IDX':temp})
                                nobs[t-1,k1,k2]=len(temp)
                                portfolio_return_EW[t-1,k1,k2]=nanmean(returns[t-lag,temp])
                                if weights != None:
                                    portfolio_return_VW[t-1, k1, k2]=nansum(returns[t-lag,temp]*weights[t-1,temp])/nansum(weights[t-1,temp])
                for k2 in range(0, K2+1):
                    if K1 > 1:
                        portfolio_return_EW[t-1, K1, k2] = portfolio_return_EW[t-1, K1 - 1, k2] - portfolio_return_EW[t-1, 0, k2]
                        if weights != None:
                            portfolio_return_VW[t-1, K1, k2] = portfolio_return_VW[t-1, K1 - 1, k2] - portfolio_return_VW[t-1, 0, k2]
                for k1 in range(0, K1+1):
                    if K2 > 1:
                        portfolio_return_EW[t-1, k1, K2] = portfolio_return_EW[t-1, k1, K2 - 1] - portfolio_return_EW[t-1, k1, 0]
                        if weights != None:
                            portfolio_return_VW[t-1, k1, K2] = portfolio_return_VW[t-1, k1, K2 - 1] - portfolio_return_VW[t-1, k1, 0]
            if time_indices is not None:
                portfolio_return_EW = portfolio_return_EW[time_indices,:,:]
                if weights != None:
                    portfolio_return_VW = portfolio_return_VW[time_indices,:,:]
                nobs=nobs[time_indices,:,:]
                bounds1=bounds1[time_indices,:]
                bounds2=bounds2[time_indices,:]
                idx=[element for element in idx if element['T'] in time_indices]
            for k1 in range(0, K1 + 1):
                for k2 in range(0, K2 + 1):
                    if k1<K1 and k2<K2:
                        nobs_avg[k1, k2] = nantstat(nobs[:, k1, k2])['mean']
                    summary_EW=nantstat(portfolio_return_EW[:, k1, k2])
                    EW_raw_mean[k1, k2] = summary_EW['mean']
                    EW_raw_tstat[k1, k2] = summary_EW['tstat']

                    if weights != None:
                        summary_VW=nantstat(portfolio_return_VW[:, k1, k2])
                        VW_raw_mean[k1, k2] = summary_VW['mean']
                        VW_raw_tstat[k1, k2] = summary_VW['tstat']


            '''The following lines are to adjust value weighted and equal weighted portfolios for risk factors'''


            results['bounds1'] = bounds1
            results['bounds2'] = bounds2
            results['idx']=idx
            results['ew_returns'] = portfolio_return_EW


            results['ew_raw_mean']=EW_raw_mean
            results['ew_raw_tstat']=EW_raw_tstat
            if weights != None:
                results['vw_returns'] = portfolio_return_VW
                results['vw_raw_mean']=VW_raw_mean
                results['vw_raw_tstat'] = VW_raw_tstat

            results['nobs'] = nobs
            results['nobs_avg']=nobs_avg
        except Exception as e:
            print(e)
        return results
    def triple_sort(self, returns, K1, K2, K3, sort_param1, sort_param2, sort_param3, weights=None, lag=0, bool_filter_indices=None, time_indices=None):
        results = dict()
        T, N = sort_param1.shape

        nobs = full([T, K1, K2, K3], nan)
        nobs_avg = full([K1, K2, K3], nan)

        portfolio_return_EW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)

        EW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1], nan)
        EW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)

        if weights != None:
            portfolio_return_VW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)
            VW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1], nan)
            VW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)


        bounds1 = full([T, K1 + 1], 1)
        bounds2 = full([T, K2 + 1], 1)
        bounds3=full([T, K3 +1], 1)

        idx=[]
        for t in range(1, T):
            q1 = nanpercentile(sort_param1[t - 1, :], linspace(0, 100, K1 + 1))
            for i, el in enumerate(q1[:-1]):
                if q1[i] >= q1[i + 1]:
                    q1[i + 1] = q1[i + 1] + 1
            q2 = nanpercentile(sort_param2[t - 1, :], linspace(0, 100, K2 + 1))
            for i, el in enumerate(q2[:-1]):
                if q2[i] >= q2[i + 1]:
                    q2[i + 1] = q2[i + 1] + 1
            q3 = nanpercentile(sort_param3[t - 1, :], linspace(0, 100, K3 + 1))
            for i, el in enumerate(q3[:-1]):
                if q3[i] >= q3[i + 1]:
                    q3[i + 1] = q3[i + 1] + 1
            bounds1[t - 1, :] = q1
            bounds2[t - 1, :] = q2
            bounds3[t - 1, :] = q3
            for k1 in range(0, K1):
                for k2 in range(0, K2):
                    for k3 in range(0, K3):
                        if k1 < K1 - 1:
                            if k2 < K2 - 1:
                                if k3< K3-1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]) & (sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] < q2[k2 + 1]) & (sort_param3[t - 1 , :] >=q3[k3]) & (sort_param3[t -1, :] < q3[k3+1]))[0]
                                elif k3 == K3-1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]) & (sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] < q2[k2 + 1]) & (sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] <=q3[k3 + 1]))[0]
                            elif k2 == K2 - 1:
                                if k3 < K3 - 1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]) & (sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] <= q2[k2 + 1]) & (sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] < q3[k3 + 1]))[0]
                                elif k3 == K3 - 1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]) & (sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] <= q2[k2 + 1]) & (sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] <= q3[k3 + 1]))[0]

                        elif k1 == K1 - 1:
                            if k2 < K2 - 1:
                                if k3< K3-1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]) &(sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] < q2[k2 + 1]) &(sort_param3[t - 1 , :] >=q3[k3]) & (sort_param3[t -1, :] < q3[k3+1]))[0]
                                elif k3 == K3-1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]) &(sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] < q2[k2 + 1]) &(sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] <=q3[k3 + 1]))[0]
                            elif k2 == K2 - 1:
                                if k3 < K3 - 1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]) &(sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] <= q2[k2 + 1]) &(sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] < q3[k3 + 1]))[0]
                                elif k3 == K3 - 1:
                                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]) &(sort_param2[t - 1, :] >= q2[k2]) & (sort_param2[t - 1, :] <= q2[k2 + 1]) &(sort_param3[t - 1, :] >= q3[k3]) & (sort_param3[t - 1, :] <= q3[k3 + 1]))[0]
                        if bool_filter_indices is not None:
                            b = where(bool_filter_indices[t - 1, :] == True)[0]
                            temp = a[in1d(a, b)]
                        else:
                            temp = a
                        temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                        idx.append({'T':t-1,'K1':k1,'K2':k2,'K3':k3,'PERMNO_IDX':temp})
                        nobs[t - 1, k1, k2 , k3] = len(temp)
                        portfolio_return_EW[t-1, k1, k2 ,k3] = nanmean(returns[t - lag, temp])
                        if weights != None:
                            portfolio_return_VW[t-1, k1, k2 ,k3] = nansum(returns[t - lag, temp] * weights[t - 1, temp]) / nansum(weights[t - 1, temp])

            for k3 in range(0, K3+1):
                for k2 in range(0, K2+1):
                    if K1 > 1:
                        portfolio_return_EW[t-1, K1, k2 ,k3] = portfolio_return_EW[t-1, K1 - 1, k2 ,k3] - portfolio_return_EW[t-1, 0, k2, k3]
                        if weights != None:
                            portfolio_return_VW[t-1, K1, k2, k3] = portfolio_return_VW[t-1, K1 - 1, k2, k3] - portfolio_return_VW[t-1, 0, k2, k3]
            for k3 in range(0, K3+1):
                for k1 in range(0, K1+1):
                    if K2 > 1:
                        portfolio_return_EW[t-1, k1, K2, k3] = portfolio_return_EW[t-1, k1, K2 - 1, k3] - portfolio_return_EW[t-1, k1, 0, k3]
                        if weights != None:
                            portfolio_return_VW[t-1, k1, K2, k3] = portfolio_return_VW[t-1, k1, K2 - 1, k3] - portfolio_return_VW[t-1, k1, 0, k3]
            for k1 in range(0,K1+1):
                for k2 in range(0, K2+1):
                    if K3>1:
                        portfolio_return_EW[t-1, k1, k2, K3] = portfolio_return_EW[t-1, k1, k2, K3-1] - portfolio_return_EW[t-1, k1, k2, 0]
                        if weights != None:
                            portfolio_return_VW[t-1, k1, k2, K3] = portfolio_return_VW[t-1, k1, k2, K3 - 1] - portfolio_return_VW[t-1, k1, k2, 0]
        if time_indices is not None:
            portfolio_return_EW = portfolio_return_EW[time_indices,:,:,:]
            if weights != None:
                portfolio_return_VW = portfolio_return_VW[time_indices,:,:,:]
            nobs=nobs[time_indices,:,:,:]
            bounds1=bounds1[time_indices,:]
            bounds2=bounds2[time_indices,:]
            bounds3=bounds3[time_indices,:]
            idx=[element for element in idx if element['T'] in time_indices]
        for k1 in range(0, K1 + 1):
            for k2 in range(0, K2 + 1):
                for k3 in range(0, K3 +1):
                    if k1 < K1 and k2 < K2 and k3<K3:
                        nobs_avg[k1, k2, k3] = nantstat(nobs[:, k1, k2,k3])['mean']
                    summary_EW = nantstat(portfolio_return_EW[:, k1, k2, k3])
                    EW_raw_mean[k1, k2, k3] = summary_EW['mean']
                    EW_raw_tstat[k1, k2, k3] = summary_EW['tstat']

                    if weights != None:
                        summary_VW = nantstat(portfolio_return_VW[:, k1, k2, k3])
                        VW_raw_mean[k1, k2, k3] = summary_VW['mean']
                        VW_raw_tstat[k1, k2, k3] = summary_VW['tstat']



        results['bounds1'] = bounds1
        results['bounds2'] = bounds2
        results['bounds3'] = bounds3
        results['idx']=idx
        results['ew_returns'] = portfolio_return_EW
        results['ew_raw_mean'] = EW_raw_mean
        results['ew_raw_tstat'] = EW_raw_tstat
        if weights != None:
            results['vw_returns'] = portfolio_return_VW
            results['vw_raw_mean'] = VW_raw_mean
            results['vw_raw_tstat'] = VW_raw_tstat

        results['nobs'] = nobs
        results['nobs_avg'] = nobs_avg

        return results
    def triple_sort_sequential(self,returns, K1, K2, K3, sort_param1, sort_param2, sort_param3, weights=None, lag=0,bool_filter_indices=None, time_indices=None):
        results = dict()
        T, N = sort_param1.shape

        nobs = full([T, K1, K2, K3], nan)
        nobs_avg = full([K1, K2, K3], nan)

        portfolio_return_EW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)
        portfolio_return_VW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)

        EW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1], nan)
        EW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)

        if weights != None:
            VW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1 ], nan)
            VW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)


        bounds1 = full([T, K1 + 1], 1)
        bounds2 = full([T, K2 + 1], 1)
        bounds3=full([T, K3 +1], 1)

        idx=[]
        for t in range(1, T):
            q1 = nanpercentile(sort_param1[t - 1, :], linspace(0, 100, K1 + 1))
            for i, el in enumerate(q1[:-1]):
                if q1[i] >= q1[i + 1]:
                    q1[i + 1] = q1[i + 1] + 1

            bounds1[t - 1, :] = q1
            for k1 in range(0, K1):
                if k1 < K1 - 1:
                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]))[0]
                elif k1 ==K1-1:
                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]))[0]
                q2 = nanpercentile(sort_param2[t - 1, a], linspace(0, 100, K2 + 1))
                for i, el in enumerate(q2[:-1]):
                    if q2[i] >= q2[i + 1]:
                        q2[i + 1] = q2[i + 1] + 1
                bounds2[t-1,:]=q2
                for k2 in range(0, K2):
                    if k2 < K2 - 1:
                        b=a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] < q2[k2 + 1]))[0]]
                    elif k2 == K2 - 1:
                        b =a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] <= q2[k2 + 1]))[0]]
                    q3 = nanpercentile(sort_param3[t - 1, b], linspace(0, 100, K3 + 1))
                    for i, el in enumerate(q3[:-1]):
                        if q3[i] >= q3[i + 1]:
                            q3[i + 1] = q3[i + 1] + 1
                    bounds3[t-1,:]=q3
                    for k3 in range(0,K3):
                        if k3 < K3 - 1:
                            c = b[where( (sort_param3[t - 1, b] >= q3[k3]) & (sort_param3[t - 1, b] < q3[k3 + 1]))[0]]
                        elif k3 == K3 - 1:
                            c = b[where((sort_param3[t - 1, b] >= q3[k3]) & (sort_param3[t - 1, b] <= q3[k3 + 1]))[0]]
                        if bool_filter_indices is not None:
                            d = where(bool_filter_indices[t - 1, :] == True)[0]
                            temp = c[in1d(c, d)]
                        else:
                            temp = c
                        temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                        idx.append({'T':t-1,'K1':k1,'K2':k2,'K3':k3,'PERMNO_IDX':temp})
                        nobs[t - 1, k1, k2 , k3] = len(temp)
                        portfolio_return_EW[t-1, k1, k2 ,k3] = nanmean(returns[t - lag, temp])
                        if weights != None:
                            portfolio_return_VW[t-1, k1, k2 ,k3] = nansum(returns[t - lag, temp] * weights[t - 1, temp]) / nansum(weights[t - 1, temp])

            for k3 in range(0, K3+1):
                for k2 in range(0, K2+1):
                    if K1 > 1:
                        portfolio_return_EW[t-1, K1, k2 ,k3] = portfolio_return_EW[t-1, K1 - 1, k2 ,k3] - portfolio_return_EW[t-1, 0, k2, k3]
                        if weights != None:
                            portfolio_return_VW[t-1, K1, k2, k3] = portfolio_return_VW[t-1, K1 - 1, k2, k3] - portfolio_return_VW[t-1, 0, k2, k3]
            for k3 in range(0, K3+1):
                for k1 in range(0, K1+1):
                    if K2 > 1:
                        portfolio_return_EW[t-1, k1, K2, k3] = portfolio_return_EW[t-1, k1, K2 - 1, k3] - portfolio_return_EW[t-1, k1, 0, k3]
                        if weights != None:
                            portfolio_return_VW[t-1, k1, K2, k3] = portfolio_return_VW[t-1, k1, K2 - 1, k3] - portfolio_return_VW[t-1, k1, 0, k3]
            for k1 in range(0,K1+1):
                for k2 in range(0, K2+1):
                    if K3>1:
                        portfolio_return_EW[t-1, k1, k2, K3] = portfolio_return_EW[t-1, k1, k2, K3-1] - portfolio_return_EW[t-1, k1, k2, 0]
                        if weights != None:
                            portfolio_return_VW[t-1, k1, k2, K3] = portfolio_return_VW[t-1, k1, k2, K3 - 1] - portfolio_return_VW[t-1, k1, k2, 0]
        if time_indices is not None:
            portfolio_return_EW = portfolio_return_EW[time_indices,:,:,:]
            if weights != None:
                portfolio_return_VW = portfolio_return_VW[time_indices,:,:,:]
            nobs=nobs[time_indices,:,:,:]
            bounds1=bounds1[time_indices,:]
            bounds2=bounds2[time_indices,:]
            bounds3=bounds3[time_indices,:]
            idx=[element for element in idx if element['T'] in time_indices]
        for k1 in range(0, K1 + 1):
            for k2 in range(0, K2 + 1):
                for k3 in range(0, K3 +1):
                    if k1 < K1 and k2 < K2 and k3<K3:
                        nobs_avg[k1, k2, k3] = nantstat(nobs[:, k1, k2,k3])['mean']
                    summary_EW = nantstat(portfolio_return_EW[:, k1, k2, k3])
                    EW_raw_mean[k1, k2, k3] = summary_EW['mean']
                    EW_raw_tstat[k1, k2, k3] = summary_EW['tstat']
                    if weights != None:
                        summary_VW = nantstat(portfolio_return_VW[:, k1, k2, k3])
                        VW_raw_mean[k1, k2, k3] = summary_VW['mean']
                        VW_raw_tstat[k1, k2, k3] = summary_VW['tstat']



        results['bounds1'] = bounds1
        results['bounds2'] = bounds2
        results['bounds3'] = bounds3
        results['idx']=idx
        results['ew_returns'] = portfolio_return_EW
        results['ew_raw_mean'] = EW_raw_mean
        results['ew_raw_tstat'] = EW_raw_tstat
        if weights != None:
            results['vw_returns'] = portfolio_return_VW
            results['vw_raw_mean'] = VW_raw_mean
            results['vw_raw_tstat'] = VW_raw_tstat

        results['nobs'] = nobs
        results['nobs_avg'] = nobs_avg

        return results
    def triple_sort_sequential_independent(self, returns, K1, K2, K3, sort_param1, sort_param2, sort_param3, weights=None, lag=0, bool_filter_indices=None, time_indices=None):
        results = dict()
        T, N = sort_param1.shape

        nobs = full([T, K1, K2, K3], nan)
        nobs_avg = full([K1, K2, K3], nan)

        portfolio_return_EW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)
        portfolio_return_VW = full([T, K1 + 1, K2 + 1, K3 + 1], nan)

        EW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1], nan)
        EW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)

        if weights != None:
            VW_raw_mean = full([K1 + 1, K2 + 1, K3 + 1 ], nan)
            VW_raw_tstat = full([K1 + 1, K2 + 1, K3 + 1], nan)


        bounds1 = full([T, K1 + 1], 1)
        bounds2 = full([T, K2 + 1], 1)
        bounds3=full([T, K3 +1], 1)

        idx=[]
        for t in range(1, T):
            q1 = nanpercentile(sort_param1[t - 1, :], linspace(0, 100, K1 + 1))
            for i, el in enumerate(q1[:-1]):
                if q1[i] >= q1[i + 1]:
                    q1[i + 1] = q1[i + 1] + 1

            bounds1[t - 1, :] = q1
            for k1 in range(0, K1):
                if k1 < K1 - 1:
                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] < q1[k1 + 1]))[0]
                elif k1 ==K1-1:
                    a = where((sort_param1[t - 1, :] >= q1[k1]) & (sort_param1[t - 1, :] <= q1[k1 + 1]))[0]
                if len(a)>0:
                    q2 = nanpercentile(sort_param2[t - 1, a], linspace(0, 100, K2 + 1))
                    q3 = nanpercentile(sort_param3[t - 1, a], linspace(0, 100, K3 + 1))
                    for i, el in enumerate(q3[:-1]):
                        if q3[i] >= q3[i + 1]:
                            q3[i + 1] = q3[i + 1] + 1
                    for i, el in enumerate(q2[:-1]):
                        if q2[i] >= q2[i + 1]:
                            q2[i + 1] = q2[i + 1] + 1
                    bounds2[t-1,:]=q2
                    bounds3[t-1,:]=q3
                    for k2 in range(0, K2):
                        for k3 in range(0,K3):
                            if k2 < K2 - 1:
                                if k3 < K3-1:
                                    b=a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] < q2[k2 + 1]) & (sort_param3[t - 1, a] >= q3[k3]) & (sort_param3[t - 1, a] < q3[k3 + 1]))[0]]
                                elif k3 == K3-1:
                                    b=a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] < q2[k2 + 1]) & (sort_param3[t - 1, a] >= q3[k3]) & (sort_param3[t - 1, a] <= q3[k3 + 1]))[0]]
                            elif k2 == K2 - 1:
                                if k3 < K3-1:
                                    b=a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] <= q2[k2 + 1])& (sort_param3[t - 1, a] >= q3[k3]) & (sort_param3[t - 1, a] < q3[k3 + 1]))[0]]
                                elif k3 == K3-1:
                                    b=a[where((sort_param2[t - 1, a] >= q2[k2]) & (sort_param2[t - 1, a] <= q2[k2 + 1])& (sort_param3[t - 1, a] >= q3[k3]) & (sort_param3[t - 1, a] <= q3[k3 + 1]))[0]]

                            if bool_filter_indices is not None:
                                d = where(bool_filter_indices[t - 1, :] == True)[0]
                                temp = b[in1d(b, d)]
                            else:
                                temp = b
                            temp=temp[where(~isnan(returns[t-lag,temp]))[0]]
                            idx.append({'T':t-1,'K1':k1,'K2':k2,'K3':k3,'PERMNO_IDX':temp})
                            nobs[t - 1, k1, k2 , k3] = len(temp)
                            portfolio_return_EW[t-1, k1, k2 ,k3] = nanmean(returns[t - lag, temp])
                            if weights != None:
                                portfolio_return_VW[t-1, k1, k2 ,k3] = nansum(returns[t - lag, temp] * weights[t - 1, temp]) / nansum(weights[t - 1, temp])

                for k3 in range(0, K3+1):
                    for k2 in range(0, K2+1):
                        if K1 > 1:
                            portfolio_return_EW[t-1, K1, k2 ,k3] = portfolio_return_EW[t-1, K1 - 1, k2 ,k3] - portfolio_return_EW[t-1, 0, k2, k3]
                            if weights != None:
                                portfolio_return_VW[t-1, K1, k2, k3] = portfolio_return_VW[t-1, K1 - 1, k2, k3] - portfolio_return_VW[t-1, 0, k2, k3]
                for k3 in range(0, K3+1):
                    for k1 in range(0, K1+1):
                        if K2 > 1:
                            portfolio_return_EW[t-1, k1, K2, k3] = portfolio_return_EW[t-1, k1, K2 - 1, k3] - portfolio_return_EW[t-1, k1, 0, k3]
                            if weights != None:
                                portfolio_return_VW[t-1, k1, K2, k3] = portfolio_return_VW[t-1, k1, K2 - 1, k3] - portfolio_return_VW[t-1, k1, 0, k3]
                for k1 in range(0,K1+1):
                    for k2 in range(0, K2+1):
                        if K3>1:
                            portfolio_return_EW[t-1, k1, k2, K3] = portfolio_return_EW[t-1, k1, k2, K3-1] - portfolio_return_EW[t-1, k1, k2, 0]
                            if weights != None:
                                portfolio_return_VW[t-1, k1, k2, K3] = portfolio_return_VW[t-1, k1, k2, K3 - 1] - portfolio_return_VW[t-1, k1, k2, 0]
        if time_indices is not None:
            portfolio_return_EW = portfolio_return_EW[time_indices,:,:,:]
            if weights != None:
                portfolio_return_VW = portfolio_return_VW[time_indices,:,:,:]
            nobs=nobs[time_indices,:,:,:]
            bounds1=bounds1[time_indices,:]
            bounds2=bounds2[time_indices,:]
            bounds3=bounds3[time_indices,:]
            idx=[element for element in idx if element['T'] in time_indices]
        for k1 in range(0, K1 + 1):
            for k2 in range(0, K2 + 1):
                for k3 in range(0, K3 +1):
                    if k1 < K1 and k2 < K2 and k3<K3:
                        nobs_avg[k1, k2, k3] = nantstat(nobs[:, k1, k2,k3])['mean']
                    summary_EW = nantstat(portfolio_return_EW[:, k1, k2, k3])
                    EW_raw_mean[k1, k2, k3] = summary_EW['mean']
                    EW_raw_tstat[k1, k2, k3] = summary_EW['tstat']

                    if weights != None:
                        summary_VW = nantstat(portfolio_return_VW[:, k1, k2, k3])
                        VW_raw_mean[k1, k2, k3] = summary_VW['mean']
                        VW_raw_tstat[k1, k2, k3] = summary_VW['tstat']
        results['bounds1'] = bounds1
        results['bounds2'] = bounds2
        results['bounds3'] = bounds3
        results['idx']=idx
        results['ew_returns'] = portfolio_return_EW
        results['ew_raw_mean'] = EW_raw_mean
        results['ew_raw_tstat'] = EW_raw_tstat
        if weights != None:
            results['vw_returns'] = portfolio_return_VW
            results['vw_raw_mean'] = VW_raw_mean
            results['vw_raw_tstat'] = VW_raw_tstat

        results['nobs'] = nobs
        results['nobs_avg'] = nobs_avg

        return results

        def output_double_sort(self,returns,K1,K2,sortParam1,sortParam2,weights,label_y_axis,label_x_axis,factors,lag=0,precision=2,addTstat=True,firm_filter=None,time_filter=None,tableNameSuffix='',sort_type='independent',outputStars=False,risk_adjustment=True,displayPercent=True):
            resultDict=dict()
            header = [label_y_axis + str(i) for i in range(1, K1 + 1)] + [label_y_axis + str(K1) + ' - ' + label_y_axis + '1']
            if time_filter is not None:
                factors2={key:val[time_filter] for key,val in factors.items()}
            else:
                factors2={key:val for key,val in factors.items()}
            if sort_type.upper()=='INDEPENDENT':
                sortedPortfolios = self.double_sort(returns, K1, K2, sortParam1, sortParam2, weights, lag, bool_filter_indices=firm_filter, time_indices=time_filter)
            elif sort_type.upper()=='SEQUENTIAL':
                sortedPortfolios=self.double_sort_sequential(returns, K1, K2, sortParam1, sortParam2, weights, lag, bool_filter_indices=firm_filter, time_indices=time_filter)
                tableNameSuffix+='_SEQUENTIAL'

            resultDict['EW_RAW_RETURNS']=sortedPortfolios['ew_returns']
            resultDict['VW_RAW_RETURNS']=sortedPortfolios['vw_returns']
            resultDict['EW_RAW_MEANS']=sortedPortfolios['ew_raw_mean']
            resultDict['EW_RAW_TSTATS']=sortedPortfolios['ew_raw_tstat']
            resultDict['VW_RAW_MEANS'] = sortedPortfolios['vw_raw_mean']
            resultDict['VW_RAW_TSTATS'] = sortedPortfolios['vw_raw_tstat']
            if not addTstat:
                resultDict['EW_RAW_TSTATS'] = array([])
                resultDict['VW_RAW_TSTATS'] = array([])
            if displayPercent:
               resultDict['EW_RAW_MEANS']=resultDict['EW_RAW_MEANS']*100
               resultDict['VW_RAW_MEANS']=resultDict['VW_RAW_MEANS']*100
            table_coef_tstatfixed(resultDict['EW_RAW_MEANS'], resultDict['EW_RAW_TSTATS'], header, precision, self.latexTablesPath + 'EW'+'_RAW_'+label_y_axis+'_by_'+label_x_axis+'_'+str(K1)+'_'+str(K2)+tableNameSuffix,addStars=outputStars)
            table_coef_tstatfixed(resultDict['VW_RAW_MEANS'], resultDict['VW_RAW_TSTATS'], header, precision,self.latexTablesPath + 'VW'+'_RAW_'+label_y_axis+'_by_'+label_x_axis+'_'+str(K1)+'_'+str(K2)+tableNameSuffix,addStars=outputStars)

            if risk_adjustment==True:
                for factor_name,factor in factors2.items():
                    ew_adjusted_portfolios=get_risk_adjusted_portfolio_returns_portfolio_returns(sortedPortfolios['ew_returns'],factor)
                    vw_adjusted_portfolios = get_risk_adjusted_portfolio_returns_portfolio_returns(sortedPortfolios['vw_returns'],factor)
                    resultDict['EW_'+factor_name+'_RETURNS']=ew_adjusted_portfolios['adj_returns']
                    resultDict['VW_' + factor_name + '_RETURNS'] = vw_adjusted_portfolios['adj_returns']
                    resultDict['EW_'+factor_name+'_MEANS'] = ew_adjusted_portfolios['alpha']
                    resultDict['EW_' + factor_name + '_TSTATS']=ew_adjusted_portfolios['tstat']
                    resultDict['VW_' + factor_name + '_MEANS'] = vw_adjusted_portfolios['alpha']
                    resultDict['VW_' + factor_name + '_TSTATS'] = vw_adjusted_portfolios['tstat']
                    if not addTstat:
                        resultDict['EW_RAW_TSTATS'] = array([])
                        resultDict['VW_RAW_TSTATS'] = array([])
                    if displayPercent:
                        resultDict['EW_'+factor_name+'_MEANS']=resultDict['EW_'+factor_name+'_MEANS']*100
                        resultDict['VW_'+factor_name+'_MEANS']=resultDict['VW_'+factor_name+'_MEANS']*100
                    table_coef_tstatfixed(resultDict['EW_'+factor_name+'_MEANS'], resultDict['EW_'+factor_name+'_TSTATS'], header, precision,self.latexTablesPath + 'EW_'+factor_name+'_'+label_y_axis+'_by_'+label_x_axis+'_'+str(K1)+'_'+str(K2)+tableNameSuffix,addStars=outputStars)
                    table_coef_tstatfixed(resultDict['VW_'+factor_name+'_MEANS'], resultDict['VW_'+factor_name+'_TSTATS'], header, precision,self.latexTablesPath + 'VW_' + factor_name + '_' + label_y_axis + '_by_' + label_x_axis + '_' + str(K1) + '_' + str(K2) + tableNameSuffix,addStars=outputStars)
            resultDict['NOBS_AVG']=sortedPortfolios['nobs_avg']
            resultDict['IDX']=sortedPortfolios['idx']
            table_coef_tstatfixed(resultDict['NOBS_AVG'],array([]),header,precision,self.latexTablesPath+'NOBS_AVG'+'_'+label_y_axis+'_by_'+label_x_axis+'_'+str(K1)+'_'+str(K2) +tableNameSuffix,addStars=outputStars)
            return resultDict
        def output_triple_sort(returns,K1,K2,K3,sortParam1,sortParam2,sortParam3,weights,label1,label2,label3,factors,lag=0,precision=2,firm_filter=None,time_filter=None,tableNameSuffix='',addTstat=True,sort_type='independent',outputStars=False,risk_adjustment=True):
            resultDict=dict()
            header = [label2 + str(i) for i in range(1, K2 + 1)] + [label2 + str(K2) + ' - ' + label2 + '1']
            if time_filter is not None:
                factors2={key:val[time_filter] for key,val in factors.items()}
            else:
                factors2={key:val for key,val in factors.items()}
            if sort_type=='independent':
                sortedPortfolios = self.triple_sort(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights,lag,bool_filter_indices=firm_filter, time_indices=time_filter)
            elif sort_type=='sequential':
                sortedPortfolios = self.triple_sort_sequential(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights, lag,bool_filter_indices=firm_filter, time_indices=time_filter)
                tableNameSuffix+='_SEQUENTIAL'
            elif sort_type=='sequential_independent':
                sortedPortfolios = self.triple_sort_sequential_independent(returns, K1, K2, K3, sortParam1, sortParam2, sortParam3, weights, lag,bool_filter_indices=firm_filter, time_indices=time_filter)
                tableNameSuffix+='_SEQUENTIAL_INDEPENDENT'

            resultDict['EW_RAW_RETURNS'] = sortedPortfolios['ew_returns']
            resultDict['VW_RAW_RETURNS'] = sortedPortfolios['vw_returns']
            resultDict['EW_RAW_MEANS'] = sortedPortfolios['ew_raw_mean']
            resultDict['EW_RAW_TSTATS'] = sortedPortfolios['ew_raw_tstat']
            resultDict['VW_RAW_MEANS'] = sortedPortfolios['vw_raw_mean']
            resultDict['VW_RAW_TSTATS'] = sortedPortfolios['vw_raw_tstat']

            table_coef_tstatfixed(np.hstack((resultDict['EW_RAW_MEANS'][0,:,:],resultDict['EW_RAW_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['EW_RAW_TSTATS'][0,:,:],resultDict['EW_RAW_TSTATS'][K1-1,:,:])), header, precision,
                                  self.latexTablesPath + 'EW' + '_RAW_' + label1 + '_by_' + label2 +'_by_'+label3+'_' + str(K1) + '_' + str(K2)+'_'+str(K3)+tableNameSuffix,addStars=outputStars)
            table_coef_tstatfixed(np.hstack((resultDict['VW_RAW_MEANS'][0,:,:],resultDict['VW_RAW_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['VW_RAW_TSTATS'][0,:,:],resultDict['VW_RAW_TSTATS'][K1-1,:,:])), header, precision,
                                  self.latexTablesPath + 'VW' + '_RAW_' + label1 + '_by_' + label2 +'_by_'+label3+'_' + str(K1) + '_' + str(K2)+'_'+str(K3)+tableNameSuffix,addStars=outputStars)
            if risk_adjustment:
                for factor_name,factor in factors2.items():
                    ew_adjusted_portfolios=get_risk_adjusted_portfolio_returns_portfolio_returns(sortedPortfolios['ew_returns'],factor)
                    vw_adjusted_portfolios = get_risk_adjusted_portfolio_returns_portfolio_returns(sortedPortfolios['vw_returns'],factor)
                    resultDict['EW_'+factor_name+'_RETURNS']=ew_adjusted_portfolios['adj_returns']
                    resultDict['VW_' + factor_name + '_RETURNS'] = vw_adjusted_portfolios['adj_returns']
                    resultDict['EW_'+factor_name+'_MEANS'] = ew_adjusted_portfolios['alpha']
                    resultDict['EW_' + factor_name + '_TSTATS']=ew_adjusted_portfolios['tstat']
                    resultDict['VW_' + factor_name + '_MEANS'] = vw_adjusted_portfolios['alpha']
                    resultDict['VW_' + factor_name + '_TSTATS'] = vw_adjusted_portfolios['tstat']
                    kk=2
                    table_coef_tstatfixed(np.hstack((resultDict['EW_'+factor_name+'_MEANS'][0,:,:],resultDict['EW_'+factor_name+'_MEANS'][K1-1,:,:]))*100,np.hstack((resultDict['EW_' + factor_name + '_TSTATS'][0,:,:],resultDict['EW_' + factor_name + '_TSTATS'][K1-1,:,:])), header, precision,self.latexTablesPath + 'EW_' +factor_name+'_'+ label1 + '_by_' + label2 +'_by_' + label3 + '_' + str(K1) + '_' + str(K2) + '_' + str(K3) +tableNameSuffix,addStars=outputStars)
                    table_coef_tstatfixed(np.hstack((resultDict['VW_'+factor_name+'_MEANS'][0,:,:],resultDict['VW_'+factor_name+'_MEANS'][K1-1,:,:]))*100, np.hstack((resultDict['VW_' + factor_name + '_TSTATS'][0,:,:],resultDict['VW_' + factor_name + '_TSTATS'][K1-1,:,:])), header, precision,self.latexTablesPath + 'VW_' +factor_name+'_'+ label1 + '_by_' + label2 +'_by_' + label3 + '_' + str(K1) + '_' + str(K2) + '_' + str(K3) +tableNameSuffix,addStars=outputStars)

            resultDict['NOBS_AVG'] = sortedPortfolios['nobs_avg']
            resultDict['IDX']=sortedPortfolios['idx']
            table_coef_tstatfixed(np.hstack((resultDict['NOBS_AVG'] [0,:,:],resultDict['NOBS_AVG'][K1-1,:,:])), array([]), header, precision,self.latexTablesPath + 'NOBS_AVG' + '_' + label1 + '_by_' + label2 +'_by_' + label3 + '_' + str( K1) + '_' + str(K2) + '_' + str(K3)+tableNameSuffix,addStars=outputStars)
            return resultDict

if __name__ == "__main__":
    pass