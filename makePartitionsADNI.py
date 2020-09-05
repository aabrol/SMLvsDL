import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

##### MMSE #####
df = pd.read_csv('./Progression_Master.csv') # Contains 828 Subject fields [sid,smriPath,age,sex,label,MMSE]

tr_smp_sizes = [428]
va_size = 200
te_size = 200
va_te_size = va_size + te_size
nReps = 20

sdir = './SampleSplits_MMSE'

try:
    os.stat(sdir)
except:
    os.mkdir(sdir)  

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        df_tr, df_te = train_test_split(df, train_size = tss, test_size=va_te_size, shuffle='True')
        df_va, df_te = train_test_split(df_te, test_size=te_size, shuffle='True')
        df_tr.to_csv(sdir+'/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_va.to_csv(sdir+'/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_te.to_csv(sdir+'/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
