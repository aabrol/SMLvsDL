import csv
import shutil
import os
import glob

with open('data/ABCD/qc_with_paths.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            src_folder = '/home/elisonj/shared/BCP/raw/BIDS_output'
            src_folder = os.path.join(src_folder, row[0], row[1], 'anat')
            mri_list = glob.glob(f'{src_folder}/*.nii.gz')
            dst = 'data-unversioned/mri'
            for src in mri_list:
                try:
                    shutil.copy(src, dst)
                except:
                    continue
