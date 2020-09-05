import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./AA_UKB.csv') # Contains 12314 Subject fields [eid,fmriPath,smriPath,age,sex,label,age5label]
# Here age sex and label are labels for the age regression, gender classification and age-and-gender-based classification tasks respectively.

tr_smp_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
va_size = int((df.shape[0] - max(tr_smp_sizes))/2) # = 1157
te_size = va_size 
va_te_size = 2*te_size
nReps = 20
#tasks = ['AgeSex','Sex','Age']
scores = ['label','sex','age5label']
taskDirs = ['./SampleSplits', './SampleSplits_Sex', './SampleSplits_Age']

for tdir, score in zipped(taskDirs,scores):

    try:
        os.stat(tdir)
    except:
        os.mkdir(tdir)   
    
    for tss in tr_smp_sizes:
        
        for rep in np.arange(nReps):
            df_tr, df_te = train_test_split(df, stratify = df[score], train_size = tss, test_size=va_te_size, shuffle='True')
            df_va, df_te = train_test_split(df_te, stratify = df_te[score], test_size=te_size, shuffle='True')
            df_tr.to_csv(tdir + '/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
            df_va.to_csv(tdir + '/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
            df_te.to_csv(tdir + '/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)

