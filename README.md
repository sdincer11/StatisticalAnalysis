# StatisticalAnalysis
 It includes all the codes to conduct portfolio sort tests and other statistical analyses.
 
 NOTES: 
 
 1) You need to upload a CSV file for monthly security data to your project folder. 
 Due to disk storage limit on GitHub, I could not upload the security file I created and called it "sample_data.gz". 
 You must upload this file to your project folder. 
 Please note that the column headers must include 
 - the security identifier (TICKER, or PERMNO, or CUSIP, etc.) -> corresponds to the parameter "idField" in PortfolioSort object   creation
 - the date identifier (YYYYMM) -> dateField
 - the return field -> the column for monthly security return
 - firm characteristics to conduct portfolio analyses with (ROE, SIZE, BOOK-TO-MARKET, IDIOSYNCRATIC VOLATILITY, etc.)
 
2) You must upload a CSV file for the returns of market risk factors (like CAPM, Fama-French 3 factors), where the date identifier
must have the same name with the date identifier of the monthly security data file i.e, YYYYMM. You can find a sample for this file among this repository's files: "sample_factors.gz".

WARNING: You need to ensure that there won't be duplicate records per each (security identifier, date identifier) combination in the monthly security data(in the bullet 1 above) you add to your project folder. Otherwise, the program will NOT run.
