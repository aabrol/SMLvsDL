import csv
import random

from os import listdir
from os.path import isfile, join, isdir

fields = ['eid', 'fmriPath', 'smriPath', 'age', 'sex', 'label', 'age5label']

abcd_path = '/home/feczk001/shared/data/ABCD/sorted/'
directories = [d for d in listdir(abcd_path) if isdir(join(abcd_path, d))]

rows = []
for directory in directories:
    if not directory.startswith('sub-'):
        continue
    anat_folder = join(abcd_path, directory, 'ses-baselineYear1Arm1/anat')
    files = [f for f in listdir(anat_folder) if isfile(join(anat_folder, f))]
    for file in files:
        if file.endswith('_T1w.nii.gz'):
            t1_file = join(anat_folder, file)
            t1_file_path = join(abcd_path, t1_file)
            t2_file = file[:-len('_T1w.nii.gz')] + '_21w.nii.gz'
            t2_file_path = join(abcd_path, t2_file)
            eid_start_index = t1_file.index('_ses') + 5
            eid_end_index = -7
            eid = t1_file[eid_start_index:eid_end_index]
            age = random.randint(0, 10)
            if random.random() < 0.5:
                sex = 'F'
            else:
                sex = 'M'
            row = [t1_file_path, t2_file_path, t1_file, sex, age, ""]
            rows.append(row)

filename = "../../data/ABCD/ABCD.csv"

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)
