
from numpy import *
import os


projectPath = os.getcwd().replace('\\','/')+'/'
matricesPath = projectPath+'Matrices/'
csvPath=projectPath+'CSV Files/'
latexPath=projectPath+'latex/'
latexTablesPath=latexPath+'tables/'
'''
    LatexGenerator helps 
'''
def table_coef_tstatfixed(c,t,names,nd,fn,cst=0,cnum=0,showTstat=True,addStars=True,hline=True):
    '''cst=args[0]
    cnum=args[1]
    %function table_coef_tstat(c,t,names,nd,fn)
    %c,t matrices
    %nd is number of digits
    %cst is a cell structure (can be multiple columns)
    %cnum is the table columns in which to put the columns i cst
    table_coef_tstat(coefs*100,tstats,names,2,'tables/table5Ara')

    %names is extended to NC number of characters by padding spaces'''
    NC=35
    names=[name+" "*(NC-len(name)) for name in names]

    if cst==0:
        cst=-1
        cnum=-1
        nn=0
    else:
        nn=length(cnum)

    with open(''.join([fn,'.tex']), 'w') as file:
        nd=str(nd)
        nc,kc=c.shape
        nt,kt=c.shape
        if t.size==0:
            for i in range(0,nc):
                m=0
                m1=0
                cs=''
                for j in range(0,kc+nn):
                    '''if j==any(cnum):
                        cs=cs+'& '+str(cst[i,m1])
                        m1=m1+1
                        j=j+1
                        c[i,:]
                    else:'''
                    cs=cs+'&$'+("{0:."+nd+"f}").format(c[i,m]).rjust(12)+'$'
                    m = m + 1
                file.write(''.join([str(names[i]),cs,'\\\\ \n']))

        else:
            for i in range(0,nc):
                if i ==nc-1 and nc!=1 and hline:
                    file.write('\hline \\ \n')
                cs=''
                ts=" "*NC
                m=0
                m1=0
                for j in range(0,kt+nn):
                    if any(cnum==j+1):
                        if isnan(cst[i,m1]):
                            cs=cs+'& -'
                            m1=m1+1
                        else:
                            cs=cs+'& '+str(cst[i,m1])
                            m1=m1+1
                            ts=ts+' & '

                    else:
                        if (type(c[i,m])==float64) | (type(c[i,m])==float) | (type(c[i,m])==float32):
                            if isnan(c[i,m]):
                                cs=cs+'& '
                            else:
                                cs=cs+'&$'+("{0:."+nd+"f}").format(c[i,m]).rjust(12)+'$'


                        elif type(c[i, m]) == str:
                            cs= cs +'& '+c[i,m]

                        if(type(c[i,m])==float64) | (type(c[i,m])==float) | (type(c[i,m])==float32):
                            if isnan(t[i,m]):
                                ts=ts+ '& '
                            elif abs(t[i,m])>=2.576:
                                if addStars:
                                    ts=ts+'&$({\\bf '+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+'}***)$'
                                else:
                                    ts=ts+'&$({\\bf '+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+'})$'
                            elif abs(t[i,m])>=1.96:
                                if addStars:
                                    ts=ts+'&$({\\bf '+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+'}**)$'
                                else:
                                    ts=ts+'&$({\\bf '+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+'})$'
                            elif abs(t[i,m])>=1.645:
                                if addStars:
                                    ts=ts+'&$({\\bf '+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+'}*)$'
                                else:
                                    ts=ts+'&$('+("{0:."+nd+"f}").format(t[i,m]).rjust(3)+')$'
                            else:
                                ts=ts+'&$('+("{0:."+nd+"f}").format(t[i,m]).rjust(10)+')$'
                        elif type(t[i, m]) == str:
                            ts = ts + t[i, m] + '& '
                        m=m+1


                file.write(''.join([names[i],cs,'\\\\[-2.5pt] \n',ts,'\\\\[0pt] \n']))
                k=2
def create_latex_tables(tablesList,tableTitle,tableLegend,panelHeaders,panelTitles=None,newTable=True,landscape=True,texFile=os.getcwd()+'latex/'+'tables.tex',startPanel='A'):
    with open(texFile,'r') as file:
        lines=file.readlines()
        count=0
        table_count=sum([count+1 if '\input{' in line else count for line in lines])
    appendLines=['\\newpage','','','','\\noindent\\begin{minipage}[t]{6.5in}\singlespace]','\\begin{small}']
    if newTable:
        appendLines.append('\\refstepcounter{table}}')

    appendLines.append('\{\\textbf{Table \\thetable'+tableTitle+'}}')
    appendLines.append('\\begin{footnotesize}')
    appendLines.append(tableLegend)
    appendLines.append('\\begin{center}')
    panelIndex=ord(startPanel)
    for tableElementsIndex,tableElements in enumerate(tablesList):
        if tableElementsIndex==0:
            appendLines.append('\\vspace{1mm}')
        else:
            appendLines.append('\\vspace{5mm}')
        panelLetter=chr(panelIndex)
        appendLines.append('\centerline{\\bf Panel '+panelLetter+': '+panelTitles[tableElementsIndex]+'}')
        appendLines.append('\\vspace{1mm}')
        #end of Latex Code for introduction
        #start of Latex Panel Specific Tabular Code
        multiColumnsSize=[]
        if type(tableElements)==list:
            if len(tableElements)>1:
                with open(latexTablesPath+tableElements[0],'r') as rootFile:
                    rootTable=rootFile.readlines()
                    multiColumnsSize.append(len(rootTable[0].split('&'))-1)
                    for rowIndex,row in enumerate(rootTable):
                        index_to_remove=row.find('\\\[')
                        if index_to_remove!=-1:
                            rootTable[rowIndex]=row[:index_to_remove]
                tables_to_merge=tableElements[1:]
                for table_to_merge in tables_to_merge:
                    with open(latexTablesPath+table_to_merge,'r') as mergeFile:
                        subTable=mergeFile.readlines()
                        multiColumnsSize.append(len(subTable[0].split('&'))-1)
                    for rowIndex,row in enumerate(subTable):
                        subTableColumns=row.split('&')
                        if len(subTableColumns)>1:
                            mergeLine='&'.join(subTableColumns[1:])
                            rootTable[rowIndex]=rootTable[rowIndex]+'&'+mergeLine

        with open(latexTablesPath+'Table'+str(table_count+1)+'.tex','w') as tempFile:
            tempFile.writelines(rootTable)
        beginTabularLine='\\begin{tabular*}{\\textwidth}'
        for cellSize in multiColumnsSize:
            beginTabularLine=beginTabularLine+'{@{\extracolsep{\\fill}}l'+''.join(['@{}r']*cellSize)+'@{\hspace{3mm}}r|'
        beginTabularLine=beginTabularLine[:-1]+'}'
        appendLines.append(beginTabularLine)
        table_count+=1
def set_latex_header():
    latex_dict = dict({
                        'OVERPRICED_DUMMY * ABHAT': '$D_{OVERPRICED_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                        'OVERPRICED_DUMMY * ABHAT * ABSVI': '$D_{OVERPRICED_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $SVI_{i,t-1}$',
                        'OVERPRICED_DUMMY * ABSVI': '$D_{OVERPRICED_{i,t-1}}$ * $SVI_{i,t-1}$',
                        'ABSVI':'$SVI_{i,t-1}$',
                        'ABHAT * ABSVI': '$\widehat{AB}_{i,t-1}$ * $SVI_{i,t-1}$',
                        'ALTMAN_DUMMY': '$D_{HIGH\_RISK_{i,t-1}}$',
                        'ALTMAN_DUMMY * BM':'$D_{HIGH\_RISK_{i,t-1}}$ * $log(BM_{i,t-1})$',
                        'ALTMAN_DUMMY * SIZE':'$D_{HIGH\_RISK_{i,t-1}}$ * $log(Size_{i,t-1})$',
                        'ALTMAN_DUMMY * DISPERSION':'$D_{HIGH\_RISK_{i,t-1}}$ * $Disp{i,t-1}$',
                        'ALTMAN_DUMMY * SUE':'$D_{HIGH\_RISK_{i,t-1}}$ * $SUE_{i,t-1}$',
                        'ALTMAN_DUMMY * CUMRET_6':'$D_{HIGH\_RISK_{i,t-1}}$ * $r_{i,t-2:t-5}$',
                        'ALTMAN_DUMMY * TURNOVER':'$D_{HIGH\_RISK_{i,t-1}}$ * $TURNOVER_{i,t-1}$',
                        'SMALL * ALTMAN_DUMMY':'$D_{HIGH\_RISK_{i,t-1}}$ * $D_{SMALL_{i,t-1}}$',
                        'SMALL * ALTMAN_DUMMY * IVOL': '$D_{SMALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'ALTMAN_DUMMY * IVOL': '$D_{HIGH\_RISK_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'LAG_TOTRET':'$REV_{i,t-1}$',
                        'IDIOSYNCRATIC_SKEWNESS':'$IDIO\_SKEW_{i,t-1}$',
                         'AMIHUD_ILLIQ': '$AMIHUD\_ILLIQ_{i,t-1}$',
                        'UNDERPRICED_DUMMY': '$D_{UNDERPRICED_{i,t-1}}$',
                        'OVERPRICED_DUMMY': '$D_{OVERPRICED_{i,t-1}}$',
                        'OVERPRICED_DUMMY * EXCESS_COVERAGE': '$D_{OVERPRICED_{i,t-1}}$ * $EXCESS\_COVERAGE_{i,t-1}$',
                        'UNDERPRICED_DUMMY * EXCESS_COVERAGE': '$D_{UNDERPRICED_{i,t-1}}$ * $EXCESS\_COVERAGE_{i,t-1}$',
                        'OVERPRICED_DUMMY * EXCESS_COVERAGE_RANK': '$D_{OVERPRICED_{i,t-1}}$ * $EXCESS\_COVERAGE\_RANK_{i,t-1}$',
                        'UNDERPRICED_DUMMY * EXCESS_COVERAGE_RANK': '$D_{UNDERPRICED_{i,t-1}}$ * $EXCESS\_COVERAGE\_RANK_{i,t-1}$',
                        'SYY_RANK':'$SYY\_RANK_{i,t-1}$',
                        'EXCESS_COVERAGE_RANK':'$MEDIA\_RANK_{i,t-1}$',
                        'SYY_RANK * EXCESS_COVERAGE_RANK':'$SYY\_RANK_{i,t-1}$ * $MEDIA\_RANK_{i,t-1}$',
                        'EXCESS_COVERAGE': '$EXCESS\_COVERAGE_{i,t-1}$',
                        'SYY * EXCESS_COVERAGE': '$SYY_{i,t-1}$ * $EXCESS\_COVERAGE_{i,t-1}$',
                        'SYY * EXCESS_COVERAGE_RANK': '$SYY_{i,t-1}$ * $MEDIA\_RANK_{i,t-1}$',
                        'MAX_RET * EXCESS_COVERAGE_RANK': '$MAX\_RET_{i,t-1}$ * $MEDIA\_RANK_{i,t-1}$',
                        'MAX_RET':'$MAX\_RET_{i,t-1}$',
                        'EXCESS_COVERAGE_DUMMY':'$D_{EXCESS\_COVERAGE_{i,t-1}}$',
                        'MAX_RET * EXCESS_COVERAGE_DUMMY': '$MAX\_RET_{i,t-1}$ * $D_{EXCESS\_COVERAGE_{i,t-1}}$',
                       'MAX_RET * NEWS':'$MAX\_RET_{i,t-1}$ * $D_{NEWS_{i,t-1}}$',
                        'SYY * EXCESS_COVERAGE_DUMMY': '$SYY_{i,t-1}$ * $D_{EXCESS\_COVERAGE_{i,t-1}}$',
                        'OVERPRICED_DUMMY * EXCESS_COVERAGE_DUMMY': '$D_{OVERPRICED_{i,t-1}}$ * $D_{EXCESS\_COVERAGE_{i,t-1}}$',
                        'UNDERPRICED_DUMMY * EXCESS_COVERAGE_DUMMY': '$D_{UNDERPRICED_{i,t-1}}$ * $D_{EXCESS\_COVERAGE_{i,t-1}}$',
                        'SMALL': '$D_{SMALL_{i,t-1}}$',
                       'SMALL * IVOL': '$D_{SMALL_{i,t-1}}$ * $IVOL_{i,t-1}$',
                       'SMALL * OPTIONALITY': '$D_{SMALL_{i,t-1}}$ * $D_{OPTION_{i,t-1}}$',
                       'IVOL * OPTIONALITY': '$IVOL_{i,t-1}$ * $D_{OPTION_{i,t-1}}$',
                       'SMALL * IVOL * OPTIONALITY': '$D_{SMALL_{i,t-1}}$ * $IVOL_{i,t-1}$ * $D_{OPTION_{i,t-1}}$',
                       'MEDIUM': '$D_{MEDIUM_{i,t-1}}$',
                       'MEDIUM * IVOL': '$D_{MEDIUM_{i,t-1}}$ * $IVOL_{i,t-1}$',
                       'MEDIUM * OPTIONALITY': '$D_{MEDIUM_{i,t-1}}$ * $D_{OPTION_{i,t-1}}$',
                       'MEDIUM * IVOL * OPTIONALITY': '$D_{MEDIUM_{i,t-1}}$ * $IVOL_{i,t-1}$ * $D_{OPTION_{i,t-1}}$',
                       'OPTIONALITY': '$D_{OPTION_{i,t-1}}$',
                       'CONSTANT': 'Constant',
                       'SIZE': '$log(Size_{i,t-1})$',
                       'PRICE': '$Price_{i,t-1}$',
                       'PRC': '$Price_{i,t-1}$',
                       'PRC_INV':'$Price^{-1}_{i,t-1}$',
                       'IVOL': '$IVOL_{i,t-1}$',
                       'SYY': '$SYY_{i,t-1}$',
                       'TURNOVER':'$TURNOVER_{i,t-1}$',
                    'VOL_RISKRANK': '$VOLRANK_{i,t-1}$',
                        'SYY * VOL_RISKRANK':'$SYY_{i,t-1}$ * $VOL_RISKRANK_{i,t-1}$',
                       'CRRANK': '$CRRANK_{i,t-1}$',
                       'SYY * CRRANK': '$SYY_{i,t-1}$ * $CRRANK_{i,t-1}$',
                       'SYY * IVOL': '$SYY_{i,t-1}$ * $IVOL_{i,t-1}$',
                       'MARKET_CAP': '$log(Size_{i,t-1})$',
                       'ME': '$log(Size_{i,t-1})$',
                       'DISPERSION': '$Disp_{i,t-1}$',
                       'SUE': '$SUE_{i,t-1}$',
                       'BM': '$log(BM_{i,t-1})$',
                       'BEME': '$log(BM_{i,t-1})$',
                       'IO': '$IO_{i,t-1}$',
                       'CUMRET_6': '$r_{i,t-2:t-5}$',
                       'ABHAT': '$\widehat{AB}_{i,t-1}$',
                       'NEWS': '$D_{NEWS_{i,t-1}}$',
                       'NEWS * ABHAT': '$D_{NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                       'NEWS * CR5': '$D_{NEWS_{i,t-1}}$ * $D_{CR5_{i,t-1}}$',
                       'NEWS * ABHAT * CR5': '$D_{NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'NEWS * OVERPRICED': '$D_{NEWS_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'NEWS * ABHAT * OVERPRICED': '$D_{NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'NO NEWS': '$D_{NO\_NEWS_{i,t-1}}$',
                       'NO NEWS * ABHAT': '$D_{NO\_NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                       'NO NEWS * CR5': '$D_{NO\_NEWS_{i,t-1}}$ * $D_{CR5_{i,t-1}}$',
                       'NO NEWS * ABHAT * CR5': '$D_{NO\_NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'CR5': '$D_{CR5_{i,t-1}}$',
                       'ABHAT * CR5': '$\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'CR5 * ABHAT': '$D_{CR5_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                       'OVERPRICED': '$D_{OVERPRICED_{i,t-1}}$',
                       'NO NEWS * OVERPRICED': '$D_{NO\_NEWS_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'NO NEWS * ABHAT * OVERPRICED':'$D_{NO\_NEWS_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'ABHAT * OVERPRICED': '$\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'NO NEWS NEXT': '$D_{NO\_NEWS_{i,t}}$',
                       'NO NEWS NEXT * ABHAT': '$D_{NO\_NEWS_{i,t}}$ * $\widehat{AB}_{i,t-1}$',
                       'NO NEWS NEXT * CR5': '$D_{NO\_NEWS_{i,t}}$ * $D_{CR5_{i,t-1}}$',
                       'NO NEWS NEXT * ABHAT * CR5': '$D_{NO\_NEWS_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'NO NEWS NEXT * OVERPRICED': '$D_{NO\_NEWS_{i,t}}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'NO NEWS NEXT * ABHAT * OVERPRICED':'$D_{NO\_NEWS_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'HIGH NEWS': '$D_{HIGH\_COVERAGE_{i,t-1}}$',
                       'HIGH NEWS * ABHAT': '$D_{HIGH\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ ',
                       'HIGH NEWS * CR5': '$D_{HIGH\_COVERAGE_{i,t-1}}$ * $D_{CR5_{i,t-1}}$ ',
                       'HIGH NEWS * ABHAT * CR5': '$D_{HIGH\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'SOME NEWS': '$D_{SOME\_COVERAGE_{i,t-1}}$',
                       'SOME NEWS * ABHAT': '$D_{SOME\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ ',
                       'SOME NEWS * CR5': '$D_{SOME\_COVERAGE_{i,t-1}}$ * $D_{CR5_{i,t-1}}$ ',
                       'SOME NEWS * ABHAT * CR5': '$D_{SOME\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'HIGH NEWS * OVERPRICED': '$D_{HIGH\_COVERAGE_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'HIGH NEWS * ABHAT * OVERPRICED': '$D_{HIGH\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'SOME NEWS * OVERPRICED': '$D_{SOME\_COVERAGE_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'SOME NEWS * ABHAT * OVERPRICED': '$D_{SOME\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                       'EXCESS NEWS': '$D_{EXCESS\_COVERAGE_{i,t-1}}$',
                       'EXCESS NEWS * ABHAT': '$D_{EXCESS\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ ',
                       'EXCESS NEWS * CR5': '$D_{EXCESS\_COVERAGE_{i,t-1}}$ * $D_{CR5_{i,t-1}}$ ',
                       'EXCESS NEWS * ABHAT * CR5': '$D_{EXCESS\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',



                       'EXCESS NEWS * OVERPRICED': '$D_{EXCESS\_COVERAGE_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'EXCESS NEWS * ABHAT * OVERPRICED': '$D_{EXCESS\_COVERAGE_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',

                       'NEGLECT NEWS': '$D_{NEGLECTED\_{i,t-1}}$',
                       'NEGLECT NEWS * ABHAT': '$D_{NEGLECTED_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ ',
                       'NEGLECT NEWS * CR5': '$D_{NEGLECTED_{i,t-1}}$ * $D_{CR5_{i,t-1}}$ ',
                       'NEGLECT NEWS * ABHAT * CR5': '$D_{NEGLECTED_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'NEGLECT NEWS * OVERPRICED': '$D_{NEGLECTED_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'NEGLECT NEWS * ABHAT * OVERPRICED': '$D_{NEGLECTED_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',

                       'NEGATIVE NEWS': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$',
                       'NEGATIVE NEWS * ABHAT': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                       'NEGATIVE NEWS * CR5': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$ * $D_{CR5_{i,t-1}}$',
                       'NEGATIVE NEWS * ABHAT * CR5': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'NEGATIVE NEWS * OVERPRICED': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'NEGATIVE NEWS * ABHAT * OVERPRICED': '$D_{NEGATIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',


                       'POSITIVE NEWS': '$D_{POSITIVE\_NEWS\_{i,t-1}}$',
                       'POSITIVE NEWS * ABHAT': '$D_{POSITIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$',
                       'POSITIVE NEWS * CR5': '$D_{POSITIVE\_NEWS\_{i,t-1}}$ * $D_{CR5_{i,t-1}}$',
                       'POSITIVE NEWS * ABHAT * CR5': '$D_{POSITIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'POSITIVE NEWS * OVERPRICED': '$D_{POSITIVE\_NEWS\_{i,t-1}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'POSITIVE NEWS * ABHAT * OVERPRICED': '$D_{POSITIVE\_NEWS\_{i,t-1}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',

                    'NEGATIVE NEWS NEXT': '$D_{NEGATIVE\_NEWS\_{i,t}}$',
                       'NEGATIVE NEWS NEXT * ABHAT': '$D_{NEGATIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$',
                       'NEGATIVE NEWS NEXT * CR5': '$D_{NEGATIVE\_NEWS\_{i,t}}$ * $D_{CR5_{i,t-1}}$',
                       'NEGATIVE NEWS NEXT * ABHAT * CR5': '$D_{NEGATIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'NEGATIVE NEWS NEXT * OVERPRICED': '$D_{NEGATIVE\_NEWS\_{i,t}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'NEGATIVE NEWS NEXT * ABHAT * OVERPRICED': '$D_{NEGATIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',


                       'POSITIVE NEWS NEXT': '$D_{POSITIVE\_NEWS\_{i,t}}$',
                       'POSITIVE NEWS NEXT * ABHAT': '$D_{POSITIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$',
                       'POSITIVE NEWS NEXT * CR5': '$D_{POSITIVE\_NEWS\_{i,t}}$ * $D_{CR5_{i,t-1}}$',
                       'POSITIVE NEWS NEXT * ABHAT * CR5': '$D_{POSITIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                       'POSITIVE NEWS NEXT * OVERPRICED': '$D_{POSITIVE\_NEWS\_{i,t}}$ * $D_{OVERPRICED_{i,t-1}}$ ',
                       'POSITIVE NEWS NEXT * ABHAT * OVERPRICED': '$D_{POSITIVE\_NEWS\_{i,t}}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                        'ESS': '$SENT_{i,t-1}$',
                        'ESS * ABHAT': '$SENT_{i,t-1}$ * $\widehat{AB}_{i,t-1}$',
                        'ESS * CR5': '$SENT_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                        'ESS * OVERPRICED': '$SENT_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                        'ESS * ABHAT * OVERPRICED': '$SENT_{i,t-1}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                        'ESS * ABHAT * CR5': '$SENT_{i,t-1}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                        'ESS NEXT': '$SENT_{i,t}$',
                        'ESS NEXT * ABHAT': '$SENT_{i,t}$ * $\widehat{AB}_{i,t-1}$',
                        'ESS NEXT * CR5': '$SENT_{i,t}$ * $D_{CR5_{i,t-1}}$',
                        'ESS NEXT * OVERPRICED': '$SENT_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                        'ESS NEXT * ABHAT * OVERPRICED': '$SENT_{i,t}$ * $\widehat{AB}_{i,t-1}$ * $D_{OVERPRICED_{i,t-1}}$',
                        'ESS NEXT * ABHAT * CR5': '$SENT_{i,t}$ * $\widehat{AB}_{i,t-1}$ * $D_{CR5_{i,t-1}}$',
                        'PROTECTIVE_CALL_DUMMY': '$D_{CALL_{i,t-1}}$',
                        'PROTECTIVE_CALL_DUMMY * ALTMAN_DUMMY':'$D_{CALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$',
                        'PROTECTIVE_CALL_DUMMY * IVOL':'$D_{CALL_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'PROTECTIVE_CALL_DUMMY * CUMRET_6':'$D_{CALL_{i,t-1}}$ * $r_{i,t-2:t-5}$',
                        'PROTECTIVE_CALL_DUMMY * SUE':'$D_{CALL_{i,t-1}}$ * $SUE_{i,t-1}$',
                        'PROTECTIVE_CALL_DUMMY * DISPERSION':'$D_{CALL_{i,t-1}}$ * $Disp{i,t-1}$',
                        'PROTECTIVE_CALL_DUMMY * ALTMAN_DUMMY * DISPERSION':'$D_{CALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $Disp{i,t-1}$',
                        'PROTECTIVE_CALL_DUMMY * ALTMAN_DUMMY * SUE':'$D_{CALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $SUE_{i,t-1}$',
                        'PROTECTIVE_CALL_DUMMY * ALTMAN_DUMMY * CUMRET_6':'$D_{CALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $r_{i,t-2:t-5}$',
                        'PROTECTIVE_CALL_DUMMY * ALTMAN_DUMMY * IVOL': '$D_{CALL_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'DOWNGRADE_DUMMY': '$D_{DOWNGRADE_{i,t-1}}$',
                        'DOWNGRADE_DUMMY * ALTMAN_DUMMY':'$D_{DOWNGRADE_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$',
                        'DOWNGRADE_DUMMY * IVOL':'$D_{DOWNGRADE_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'DOWNGRADE_DUMMY * CUMRET_6':'$D_{DOWNGRADE_{i,t-1}}$ * $r_{i,t-2:t-5}$',
                        'DOWNGRADE_DUMMY * SUE':'$D_{DOWNGRADE_{i,t-1}}$ * $SUE_{i,t-1}$',
                        'DOWNGRADE_DUMMY * DISPERSION':'$D_{DOWNGRADE_{i,t-1}}$ * $Disp{i,t-1}$',
                        'DOWNGRADE_DUMMY * ALTMAN_DUMMY * DISPERSION':'$D_{DOWNGRADE_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $Disp{i,t-1}$',
                        'DOWNGRADE_DUMMY * ALTMAN_DUMMY * SUE':'$D_{DOWNGRADE_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $SUE_{i,t-1}$',
                        'DOWNGRADE_DUMMY * ALTMAN_DUMMY * CUMRET_6':'$D_{DOWNGRADE_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $r_{i,t-2:t-5}$',
                        'DOWNGRADE_DUMMY * ALTMAN_DUMMY * IVOL': '$D_{DOWNGRADE_{i,t-1}}$ * $D_{HIGH\_RISK_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'OVERPRICED_DUMMY * IVOL':'$D_{OVERPRICED_{i,t-1}}$ * $IVOL_{i,t-1}}$',
                        'OVERPRICED_DUMMY * LPM_TAIL_RISK':'$D_{OVERPRICED_{i,t-1}}$ * $LPM_{i,t-1}}$',
                        'IVOL * LPM_TAIL_RISK': '$IVOL_{i,t-1}}$ * $LPM_{i,t-1}$',
                        'OVERPRICED_DUMMY * IVOL * LPM_TAIL_RISK':'$D_{OVERPRICED_{i,t-1}}$ * $IVOL_{i,t-1}}$ * $LPM_{i,t-1}$',
                        'INDUSTRY_DUMMY': 'Industry Fixed Effects',
                        'ILLIQ': '$ILLIQ_{i,t-1}$',
                        'CVILLIQ':'$CVILLIQ_{i,t-1}$',
                        'CVTURN':'$CVTURN_{i,t-1}$',
                        'ILLIQ36': '$ILLIQ36_{i,t-1}$',
                        'CVILLIQ36':'$CVILLIQ36_{i,t-1}$',
                        'CVTURN36':'$CVTURN36_{i,t-1}$',

                        'JUMP':'$JUMP_{i,t-1}$',
                        'VRP':'$VRP_{i,t-1}$',
                        'MOM':'$MOM_{i,t-1}$',
                        'RELATIVE SPREAD':'$OPTION\_SPREAD_{i,t-1}$',
                        'MONEYNESS':'$S/K_{i,t-1}$',
                        'GAMMA':'$\gamma_{i,t-1}$',
                        'IVOL * CVTURN':'$IVOL_{i,t-1}$ * $CVTURN_{i,t-1}$',
                        'LEFT TAIL RISK':'$TAIL_{i,t-1}$',

                        'BIG':'$D_{BIG_{i,t-1}}$',
                        'HIGH IVOL':'$D_{HIGH\_IVOL_{i,t-1}}$',
                        'LOW IVOL':'$D_{LOW\_IVOL_{i,t-1}}$',
                        'HIGH ILLIQ':'$D_{HIGH\_ILLIQ_{i,t-1}}$',
                        'LOW ILLIQ':'$D_{LOW\_ILLIQ_{i,t-1}}$',
                        'HIGH ILLIQ36':'$D_{HIGH\_ILLIQ36_{i,t-1}}$',
                        'LOW ILLIQ36':'$D_{LOW\_ILLIQ36_{i,t-1}}$',

                        'BIG * CVTURN': '$D_{BIG_{i,t-1}}$ * $CVTURN_{i,t-1}$',
                        'SMALL * CVTURN': '$D_{SMALL_{i,t-1}}$ * $CVTURN_{i,t-1}$',
                        'HIGH IVOL * CVTURN': '$D_{HIGH\_IVOL_{i,t-1}}$ * $CVTURN_{i,t-1}$',
                        'LOW IVOL * CVTURN': '$D_{LOW\_IVOL_{i,t-1}}$ * $CVTURN_{i,t-1}$',
                        'HIGH ILLIQ * TURNOVER': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $TURNOVER_{i,t-1}$',
                        'HIGH ILLIQ * CVTURN': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $CVTURN_{i,t-1}$',
                        'HIGH ILLIQ * CVTURN36': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'LOW ILLIQ * CVTURN': '$D_{LOW\_ILLIQ_{i,t-1}}$ * $CVTURN_{i,t-1}$',

                        'BIG * CVTURN36': '$D_{BIG_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'SMALL * CVTURN36': '$D_{SMALL_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'HIGH IVOL * CVTURN36': '$D_{HIGH\_IVOL_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'LOW IVOL * CVTURN36': '$D_{LOW\_IVOL_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'HIGH ILLIQ36 * CVTURN36': '$D_{HIGH\_ILLIQ36_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'LOW ILLIQ36 * CVTURN36': '$D_{LOW\_ILLIQ36_{i,t-1}}$ * $CVTURN36_{i,t-1}$',

                        'BIG * CVILLIQ': '$D_{BIG_{i,t-1}}$ * $CVILLIQ_{i,t-1}$',
                        'SMALL * CVILLIQ': '$D_{SMALL_{i,t-1}}$ * $CVILLIQ_{i,t-1}$',
                        'HIGH IVOL * CVILLIQ': '$D_{HIGH\_IVOL_{i,t-1}}$ * $CVILLIQ_{i,t-1}$',
                        'LOW IVOL * CVILLIQ': '$D_{LOW\_IVOL_{i,t-1}}$ * $CVILLIQ_{i,t-1}$',
                        'HIGH ILLIQ * CVILLIQ36': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'HIGH ILLIQ * CVTURN36': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $CVTURN36_{i,t-1}$',
                        'HIGH ILLIQ * IVOL': '$D_{HIGH\_ILLIQ_{i,t-1}}$ * $IVOL_{i,t-1}$',
                        'LOW ILLIQ * CVILLIQ': '$D_{LOW\_ILLIQ_{i,t-1}}$ * $CVILLIQ_{i,t-1}$',

                        'BIG * CVILLIQ36': '$D_{BIG_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'SMALL * CVILLIQ36': '$D_{SMALL_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'HIGH IVOL * CVILLIQ36': '$D_{HIGH\_IVOL_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'LOW IVOL * CVILLIQ36': '$D_{LOW\_IVOL_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'HIGH ILLIQ36 * CVILLIQ36': '$D_{HIGH\_ILLIQ36_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',
                        'LOW ILLIQ36 * CVILLIQ36': '$D_{LOW\_ILLIQ36_{i,t-1}}$ * $CVILLIQ36_{i,t-1}$',

                        'OPEN INTEREST':'$OPENINT_{i,t-1}$',
                        'HIGH AMIHUDVOL':'$D_{AMIHUDVOL_{i,t-1}}$',
                        'HIGH CVTURN':'$D_{TURNVOL_{i,t-1}}$',
                        'ILLIQ * CVILLIQ':'$ILLIQ_{i,t-1}$ * $CVILLIQ_{i,t-1}$',
                        'ILLIQ * CVTURN':'$ILLIQ_{i,t-1}$ * $CVTURN_{i,t-1}$',
                        'ILLIQ36 * CVILLIQ36':'$ILLIQ36_{i,t-1}$ * $CVILLIQ36_{i,t-1}$',
                        'ILLIQ36 * CVTURN36':'$ILLIQ36_{i,t-1}$ * $CVTURN36_{i,t-1}$',
        'NUMEST': '$log(Analyst Coverage_{i,t-1})$',
        'SYSVOL': '$SysVOL_{i,t-1}$',
        'AMIHUDVOL':'$AMIHUDVOL_{i,t-1}$',
        'CVTURN60':'$CVTURN60_{i,t-1}$',
        'CVILLIQ60':'$CVILLIQ60_{i,t-1}$',
        'DTURNVOL' : '$DTURNVOL_{i,t-1}$',
        'TURNVOL': '$TURNVOL_{i,t-1}$',
        'VOV':'$VOV_{i,t-1}$',
        'TOTVOL CHANGE PAST6':'$\Delta VOL_{i,t-1}$',
        'CONTEMP IMPLIED VOL CHANGE':'$ln(IV_{i,t}/IV_{i,t-1}$',
        'OPTION DEMAND':'($Open Int_{i,t-1}$/$Stock Vol_{i,t-1}$)*$10^{3}$','Intercept':'$Constant$'
                       })
    return latex_dict

class SubPanel():
    def __init__(self,tables,subPanelHeaders=[],useSameHeader=False,useTableName=False):
        self.tables=tables
        self.useTableName=useTableName
        if type(subPanelHeaders)==str:
            self.subPanelHeaders=list(subPanelHeaders)
        elif len(subPanelHeaders)==0:
            self.subPanelHeaders=['']*len(self.tables)
        else:
            self.subPanelHeaders=subPanelHeaders
        self.config_file=latexTablesPath+'tables_lookup.txt'
        self.tableID=self.get_table_id()
        self.multiColumns=self.create_table()
    def create_table(self):
        multiColumns=[]
        if type(self.tables)==list:
            if len(self.tables)>1:
                with open(latexTablesPath+self.tables[0],'r') as rootFile:
                    rootTable=rootFile.readlines()
                    multiColumns.append(len(rootTable[0].split('&')))
                    for rowIndex,row in enumerate(rootTable):
                        index_to_remove=row.find('\\\[')
                        if index_to_remove!=-1:
                            rootTable[rowIndex]=row[:index_to_remove]
                tables_to_merge=self.tables[1:]
                for table_to_merge in tables_to_merge:
                    with open(latexTablesPath+table_to_merge,'r') as mergeFile:
                        subTable=mergeFile.readlines()
                        multiColumns.append(len(subTable[0].split('&'))-1)
                    for rowIndex,row in enumerate(subTable):
                        subTableColumns=row.split('&')
                        if len(subTableColumns)>1:
                            mergeLine='&'.join(subTableColumns[1:])
                            rootTable[rowIndex]=rootTable[rowIndex]+'&'+mergeLine
            else:
                with open(latexTablesPath+self.tables[0],'r') as rootFile:
                    rootTable=rootFile.readlines()
                    first_ampersand_line=[line for line_idx,line in enumerate(rootTable) if '&' in line][0]
                    multiColumns.append(len(first_ampersand_line.split('&')))
        elif type(self.tables)==str:
            with open(latexTablesPath+self.tables,'r') as rootFile:
                rootTable=rootFile.readlines()
                multiColumns.append(len(rootTable[0].split('&')))


        with open(latexTablesPath+'Table'+str(self.tableID)+'.tex','w') as tempFile:
            tempFile.writelines(rootTable)
        return multiColumns

    def get_table_id(self):
        if not self.useTableName:
            tableID=0
            tables=':'.join(self.tables)
            found=False
            if not os.path.exists(self.config_file):
                with open(self.config_file,'w') as file:
                    file.write(','.join([str(tableID),tables,datetime.datetime.now().strftime('%d %b %Y %a %H:%M:%S')])+'\n')
            else:
                with open(self.config_file,'r') as file:
                    last_id=max([int(line.split(',')[0]) for line in file.readlines()])
                    tableID=last_id+1
                '''
                records=[record for record in file.readlines()]
                for record in records:
                    tup=record.split(',')
                    id=int(tup[0])
                    if tables==tup[1]:
                        tableID=id
                        #To assign a new ID for each run, I changed found=True to found=False
                        found=False
                        break
                if len(records)>0:
                    if found==False:
                        tableID=id+1'''

                with open(self.config_file,'a') as file:
                    file.write(','.join([str(tableID),tables,datetime.datetime.now().strftime('%d %b %Y %a %H:%M:%S')])+'\n')
        else:
            with open(self.config_file,'w') as file:
                file.write(','.join([str(tableID),tables,datetime.datetime.now().strftime('%d %b %Y %a %H:%M:%S')])+'\n')

        return tableID

class Panel():
    def __init__(self,subPanels=[],panelTitle='',panelHeaders=[]):
        self.subPanels=subPanels
        self.countSubPanels=len(subPanels)
        self.multiColumns=self.getMultiColumns()
        self.countColumns=sum(self.multiColumns)
        self.panelHeaders=panelHeaders
        self.panelTitle=panelTitle
        self.latexCode=self.setLatex()

    def getMultiColumns(self):
        multiColumns=[]
        if self.countSubPanels>0:
            multiColumns=self.subPanels[0].multiColumns
        return multiColumns
    def setLatex(self):
        latexCodeLines=[]
        if len(self.subPanels)>0:
            for subPanelIndex,subPanel in enumerate(self.subPanels):
                if subPanelIndex==0:
                    latexCodeLines.append('{\\bf{'+self.panelTitle+'}}')
                    latexCodeLines.append('\\vspace{5mm} \\\\')
                else:
                    latexCodeLines.append('\\vspace{2mm} \\\\')
                if subPanelIndex==0:
                    beginTabular='\\begin{tabular*}{\\linewidth}{@{\extracolsep{\\fill}}l'
                    multiColumnsSize=[width-1 if index==0 else width for index, width in enumerate(self.multiColumns)]
                    for cellSize in multiColumnsSize:
                        beginTabular=beginTabular+''.join(['@{}r']*(cellSize-1))+'@{\hspace{3mm}}r|'
                    beginTabular=beginTabular[:-1]+'}'
                    latexCodeLines.append(beginTabular)
                latexCodeLines.append('\hline')
                if subPanelIndex==0:
                    for headerRow in self.panelHeaders:
                        temp=[]
                        prevStartColumn=0
                        prevEndColumn=0
                        for cell in headerRow:
                            if cell[1][0]-prevStartColumn>0:
                                width=cell[1][0]-prevStartColumn
                                temp.append('\multicolumn{'+str(width)+'}{c}{}')

                            else:
                                text=cell[0]
                                width=cell[1][1]-cell[1][0]
                                if width>1:
                                    if type(text)==str:
                                        temp.append('\multicolumn{'+str(width)+'}{c}{'+text+'}')
                                    elif type(text)==list:
                                        temp.append(' & '.join(text))
                                elif width==1:
                                    if type(text)==str:
                                        temp.append(text)
                                    elif type(text)==list:
                                        temp.append(text[0])
                            prevStartColumn=cell[1][1]
                        if cell[1][1]-self.countColumns>0:
                            width=cell[1][0]-prevStartColumn
                            temp.append('\multicolumn{'+str(width)+'}{c}{}')
                        latexCodeLines.append(' & '.join(temp)+' \\\\')
                    latexCodeLines.append('\hspace{0.5mm} \\\\ \hline')

                if len([header for header in subPanel.subPanelHeaders if len(header)>0]):

                    tempColumns=copy.deepcopy(subPanel.multiColumns)
                    subPanelName=''.join(['&']*(self.countColumns-len(tempColumns)))
                    tempColumns.insert(0,0)
                    aa=subPanelName.split('&')
                    columnIndex=0
                    for index,columnSize in enumerate(tempColumns[:-1]):
                        columnIndex+=columnSize
                        aa.insert(columnIndex,subPanel.subPanelHeaders[index])
                    sil_gitsin=[aa[0]]+[ '&'+column for column in aa[1:-1]]
                    bb=' '.join(sil_gitsin)+' \\\\'
                    latexCodeLines.append(bb)
                latexCodeLines.append('\input{Table'+str(subPanel.tableID)+'}')
                latexCodeLines.append('\hspace{1mm} \\\\')
            latexCodeLines.append('\end{tabular*}')

        return latexCodeLines
    def addSubPanel(self,subPanel):
        self.subPanels.append(subPanel)
        self.countSubPanels+=1
    def updateHeaders(self,panelHeaders):
        self.panelHeaders=panelHeaders
    def deleteSubPanel(self,index):
        self.subPanels=[subPanel for panelIdx,subPanel in enumerate(self.SubPanels) if panelIdx!=index]
        self.countSubPanels-=1

class LatexTable():
    def __init__(self,panels=[],legend='',title='',fontSize='small',continuedTable=False,vspace='5mm',appendTo=latexTablesPath+'tables.tex',orientation='portrait'):
        self.panels=panels
        self.tableIDs=self.getTableIDs()
        self.legend=legend
        self.title=title
        self.fontSize=fontSize
        self.continuedTable=continuedTable
        self.countPanels=len(self.panels)
        self.vspace=vspace
        self.appendTo=appendTo
        self.orientation=orientation
        if self.countPanels>0:
            self.latexCode=self.setLatex()
        else:
            self.latexCode=''

        aa=2
    def getTableIDs(self):
        tableIDs=[]
        for panel in self.panels:
            for subpanel in panel.subPanels:
                tableIDs.append(subpanel.tableID)
        return tableIDs

    def addPanel(self,Panels):
        if type(Panels)==list:
            self.panels.extend(Panels)
            self.countPanels+=len(Panels)
        elif type(Panels)==Panel:
            self.panels.append([Panels])
            self.countPanels+=1
    def deletePanel(self,position):
        if self.countPanels>0:
            self.panels=[panel for panelIndex,panel in enumerate(self.panels) if panelIndex!=position]
            self.countPanels-=1
    def setLatex(self):
        if not os.path.exists(latexPath+'last_latex_table_counter.txt'):
                latex_id=1
        else:
            with open(latexPath+'last_latex_table_counter.txt','r') as f:
                latex_id=int([line for line in f.readlines()][0])+1
        latexCodeLines=[]
        if self.orientation=='landscape':
            latexCodeLines.append('\\begin{landscape}')
        latexCodeLines.append('\\noindent\\begin{table}[t]\\begin{minipage}[t]{6.5in}\singlespace')
        latexCodeLines.append('\\begin{'+self.fontSize+'}')
        if self.continuedTable==False:
            latexCodeLines.append('{\\refstepcounter{table}}')
        latexCodeLines.append('\phantomsection {\label{'+str(latex_id)+'}}')
        latexCodeLines.append('{\\textbf{Table \\thetable - '+self.title+'}} \\\\')
        latexCodeLines.append('\\begin{footnotesize}')
        latexCodeLines.append(self.legend)
        latexCodeLines.append('\\begin{center}')
        for panel in self.panels:
            for latexLine in panel.latexCode:
                latexCodeLines.append(latexLine)
            latexCodeLines.append('\\vspace{'+self.vspace+'}')
        latexCodeLines=latexCodeLines[:-1]
        latexCodeLines.append('\end{center}')
        latexCodeLines.append('\end{footnotesize}')
        latexCodeLines.append('\end{small}')
        latexCodeLines.append('\end{minipage}')
        latexCodeLines.append('\end{table}')
        if self.orientation=='landscape':
            latexCodeLines.append('\\end{landscape}')
        with open(latexPath+'last_latex_table_counter.txt','w') as f:
            f.write(str(latex_id))
        return latexCodeLines
    def attachTable(self):
        tableStartIndices=[]
        tableEndIndices=[]
        linesToRemove=[]
        tableExists=False
        with open(self.appendTo,'r') as file:
            lines=[line.replace('\n','') for line in file.readlines()]
        if len(lines)>0:
            for idx,line in enumerate(lines):
                if 'begin{minipage}' in line:
                    tableStartIndices.append(idx)
                if 'end{minipage}' in line:
                    tableEndIndices.append(idx)

            tableIDSet=set(self.tableIDs)
            tablesFullMatched=[]
            for tableIDX,startIndex in enumerate(tableStartIndices):
                tablesMatched=[]
                tableLines=lines[tableStartIndices[tableIDX]:tableEndIndices[tableIDX]+1]
                for tableLine in tableLines:
                    for tableID in self.tableIDs:
                        if 'input{' in tableLine and str(tableID)+'}' in tableLine:
                            tablesMatched.append(tableID)
                if len(tablesMatched)>0:
                    if tableIDSet==set(tablesMatched):
                        tablesFullMatched.append(tableIDX)
            if len(tablesFullMatched)>0:
                tableExists=True
        if tableExists==False:
            if len(lines)>0:
                with open(self.appendTo,'a') as file:
                    [file.write('\n') for i in range(0,6)]
                    [file.write(line+'\n') for line in self.latexCode]

            else:
                with open(self.appendTo,'a') as file:
                    [file.write(line+'\n') for line in self.latexCode]

        k=2

    @staticmethod
    def deleteTable_modificationTime(withinTimePeriod=1,timeUnit='seconds',tablesFolder=latexTablesPath,prefix='Table',suffix='.tex',tablesFile=latexPath+'tables.tex',tablesLookupFile=latexTablesPath+'tables_lookup.txt'):
        if timeUnit=='minutes':
            withinTimePeriod*=60
        elif timeUnit=='hours':
            withinTimePeriod*=3600
        elif timeUnit=='days':
            withinTimePeriod=withinTimePeriod*3600*24
        elif timeUnit=='weeks':
            withinTimePeriod=withinTimePeriod*3600*24*7
        elif timeUnit=='months':
            withinTimePeriod=withinTimePeriod*3600*24*7*4
        now=time.time()
        all=os.listdir(tablesFolder)
        filesOnly=[item for item in all if '.' in item]
        filesTarget=[file for file in filesOnly if file.endswith(suffix) and file.startswith(prefix)]
        filesToRemove=[]
        IDsToRemove=[]
        fileNamesToRemove=[]
        for file in filesTarget:
            try:
                if now-os.path.getmtime(tablesFolder+file)<=withinTimePeriod:
                    filesToRemove.extend([file])
                    IDsToRemove.extend([file.replace(prefix,'').replace(suffix,'').replace('.','')])
                    fileNamesToRemove.extend([file.replace(suffix,'')])
            except Exception as e:
                print(e)
                continue
        #The following lines are to exclude tables from the TEX File including the LATEX tables
        if os.path.exists(tablesFile):
            with open(tablesFile,'r') as latexTableFile:
                lines=latexTableFile.readlines()
            tableStartIndices=[]
            tableEndIndices=[]
            linesToRemove=[]
            for idx,line in enumerate(lines):
                if 'begin{minipage}' in line:
                    tableStartIndices.append(idx)
                if 'end{minipage}' in line:
                    tableEndIndices.append(idx)
            for tableIDX,startIndex in enumerate(tableStartIndices):
                tableLines=lines[tableStartIndices[tableIDX]:tableEndIndices[tableIDX]+1]
                for tableLine in tableLines:
                    for tableName in fileNamesToRemove:
                        if 'input{' in tableLine and tableName+'}' in tableLine:
                            linesToRemove.extend(list(range(tableStartIndices[tableIDX],tableEndIndices[tableIDX]+1)))


            linesToRemove=list(sort(list(set(linesToRemove))))
            lines=[line for idx,line in enumerate(lines) if idx not in linesToRemove]
            indices=[id for id,line in enumerate(lines) if line!='\n']
            if len(indices)>0:
                with open(tablesFile,'w') as file:
                    firstLine=min(indices)
                    file.writelines(lines[firstLine:])
            else:
                with open(tablesFile,'w') as file:
                    k=2

        if os.path.exists(tablesLookupFile):
        #The following lines are to exclude the table records from the TABLE LOOKUP file
            with open(tablesLookupFile,'r') as lookup:
                lines=lookup.readlines()
            filteredLines=[line for line in lines if line.split(',')[0] not in IDsToRemove]
            with open(tablesLookupFile,'w') as lookup:
                lookup.writelines(filteredLines)

        [os.remove(tablesFolder+file) for file in filesToRemove]
    @staticmethod
    def deleteTable_tableNumber(tableNumbers=[],tablesFolder=latexTablesPath,prefix='Table',suffix='.tex',tablesFile=latexPath+'tables.tex',tablesLookupFile=latexTablesPath+'tables_lookup.txt'):

        #The following lines are to exclude tables from the TEX File including the LATEX tables
        if os.path.exists(tablesFile):
            with open(tablesFile,'r') as latexTableFile:
                lines=latexTableFile.readlines()
            if len(lines)>0:
                tableNumbers=[number-1 for number in tableNumbers]
                tableStartIndices=[]
                tableEndIndices=[]
                linesToRemove=[]
                for idx,line in enumerate(lines):
                    if 'begin{minipage}' in line:
                        tableStartIndices.append(idx)
                    if 'end{minipage}' in line:
                        tableEndIndices.append(idx)
                for tableIDX,startIndex in enumerate(tableStartIndices):
                    if tableIDX in tableNumbers:
                        linesToRemove.extend(list(range(tableStartIndices[tableIDX],tableEndIndices[tableIDX]+1)))



                linesToRemove=list(sort(list(set(linesToRemove))))
                lines=[line for idx,line in enumerate(lines) if idx not in linesToRemove]
                indices=[id for id,line in enumerate(lines) if line!='\n']
                if len(indices)>0:
                    with open(tablesFile,'w') as file:
                        firstLine=min(indices)
                        file.writelines(lines[firstLine:])
                else:
                    with open(tablesFile,'w') as file:
                        k=2

if __name__ == "__main__":
    pass