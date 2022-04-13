import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import nibabel as nib

df = pd.read_csv('../data/ABCD/qc_with_paths.csv')

def prepend_path(smri_path):
    prefix = '/home/elisonj/shared/BCP/raw/BIDS_output'
    return os.path.join(prefix, smri_path)
df['smriPath'] = df['smriPath'].apply(prepend_path)

df['width'] = pd.Series(dtype='int')
df['height'] = pd.Series(dtype='int')
df['depth'] = pd.Series(dtype='int')

data_shape_to_count = {}
for index, row in df.iterrows():
    smri_path = row["smriPath"]
    img = nib.load(smri_path)
    header = img.header
    shape = header.get_data_shape()
    if shape not in data_shape_to_count.keys():
        data_shape_to_count[shape] = 1
    else:
        data_shape_to_count[shape] += 1
    df.at[index, 'width'] = shape[0]
    df.at[index, 'height'] = shape[1]
    df.at[index, 'depth'] = shape[2]

max_value = max(data_shape_to_count, key=data_shape_to_count.get)

df = df[(df.width == max_value[0]) & (df.height == max_value[1]) & (df.depth == max_value[2])]

n = len(df.index)
tr_size = int(round(0.8 * n))
tr_smp_sizes = [tr_size]
va_size = int(round(0.1 * n))
te_size = n - (tr_size + va_size)
va_te_size = va_size + te_size
nReps = 20

sdir = './SampleSplits_MMSE'

isdir = os.path.isdir(sdir)

if not isdir:
    os.mkdir(sdir)
os.stat(sdir)

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        df_tr, df_te = train_test_split(df, train_size=tss, test_size=va_te_size, shuffle='True')
        df_va, df_te = train_test_split(df_te, test_size=te_size, shuffle='True')
        df_tr.to_csv(sdir + '/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_va.to_csv(sdir + '/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_te.to_csv(sdir + '/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
