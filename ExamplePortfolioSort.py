from PortfolioSort import *
import os
sorter = PortfolioSort(csvFile = os.getcwd() + '\\sample_data.gz',
              factorsFile = os.getcwd() + '\\sample_factors.gz',
              variablesList= ['SIZE','BEME','IVOL','SUE','DISPERSION','CUMRET_6'],
              tstart=197001, tend= 201612,
              returnField='TOTRET', dateField='YYYYMM', idField='PERMNO', portfolioWeightField='SIZE')
sorter.output_double_sort(returns= sorter.matrices['TOTRET'],K1=5,K2=5,
                          sortParam1=sorter.matrices['SIZE'],sortParam2= sorter.matrices['BEME'],
                          label_y_axis='SIZE',label_x_axis='BM',
                          factors = sorter.factors, weights = None,
                          lag=0, precision=2,
                          sort_type='independent')
k=2