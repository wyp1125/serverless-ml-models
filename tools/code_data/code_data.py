import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
parser = argparse.ArgumentParser(description='This program takes in the output files of read_data.py, removes outliers,  and code the data')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix for input files")
parser.add_argument('-o', '--outprefix', type=str, required=True, help="prefix for output files")
parser.add_argument('-r', '--rmoutlier', action='store_true', help="remove outliers")
args = parser.parse_args()

c_outlier_cut=0.03
iqr_magnitude=1.5

rmol=args.rmoutlier
vardesfile=args.inprefix+".vardes"
if not os.path.isfile(vardesfile):
    print("Cannot find the .vardes file!")
    exit()
rawxfile=args.inprefix+".rawx"
if not os.path.isfile(rawxfile):
    print("Cannot find the .rawx file!")
    exit()
rawyfile=args.inprefix+".rawy"
if not os.path.isfile(rawyfile):
    print("Cannot find the .rawy file!")
    exit()
vardes=pd.read_csv(vardesfile,sep='\t')
n=len(vardes.index)
vardict={}
for i in range(0,n):
    vardict[vardes.iloc[i,0]]=vardes.iloc[i,4]
rawx=pd.read_csv(rawxfile)
m=len(rawx.index)
dumbvar=[]
dummvar=[]
numvar=[]
for colname in rawx.columns:
    if vardict[colname]=='cb':
        dumbvar.append(colname)
    if vardict[colname]=='cm':
        dummvar.append(colname)
    if vardict[colname]=='n':
        numvar.append(colname)
tempx=pd.get_dummies(rawx,columns=dumbvar,drop_first=True)
codedx=pd.get_dummies(tempx,columns=dummvar)

if rmol:
    keepfea=[]
    for feaname in codedx.columns:
        if feaname in numvar:
            q4=np.max(codedx[feaname])
            q3=np.quantile(codedx[feaname],0.75)
            q1=np.quantile(codedx[feaname],0.25)
            q0=np.min(codedx[feaname])
            adjfea=codedx[feaname]
            if q3-q1>0:
                iqr=q3-q1
                upperlimit=q3+iqr_magnitude*iqr
                lowerlimit=q1-iqr_magnitude*iqr
                for i in range(m):
                    if codedx.loc[i,feaname]>upperlimit:
                        codedx.loc[i,feaname]=upperlimit
                    if codedx.loc[i,feaname]<lowerlimit:
                        codedx.loc[i,feaname]=lowerlimit
            keepfea.append(feaname)
        else:
            varstat=codedx[feaname].value_counts()
            frac=varstat[0]/(varstat[0]+varstat[1])
            if frac>c_outlier_cut and frac<1-c_outlier_cut:
                keepfea.append(feaname)
    codedx_ol_rmd=codedx[keepfea]
else:
    codedx_ol_rmd=codedx.copy()
scaler = MinMaxScaler()
scaler.fit(codedx_ol_rmd.to_numpy())
codedx_final=pd.DataFrame(data=scaler.transform(codedx_ol_rmd.to_numpy()),columns=codedx_ol_rmd.columns)
#print(codedx_final)

rawy=pd.read_csv(rawyfile)
if vardict[rawy.columns[0]][0:2]=='yc':
    codedy=pd.get_dummies(rawy)
else:
    codedy=rawy.copy()
codedy.to_csv(args.outprefix+".codedy",index=False)
codedx_final.to_csv(args.outprefix+".codedx",index=False)
