import csv

from os import listdir
from os.path import isfile, join

fields = ['t1path', 't2path', 'target']

my_path = '/home/feczk001/shared/data/nnUNet/raw_data/Task516_525/images/'
only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

rows = []
for i in range(0, len(only_files) // 2):
    t1_file = only_files[2 * i]
    row = [join(my_path, t1_file), join(my_path, only_files[2 * i + 1]), int(t1_file[0])]
    rows.append(row)

filename = "../../data/ABCD/ABCD.csv"

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)
