import os
import nibabel as nib

directory = '/home/miran045/reine097/projects/AlexNet_Abrol2021/data-unversioned/abcd'

data_shape_to_count = {}
for filename in os.listdir(directory):
	f = os.path.join(directory, filename)
	if os.path.isfile(f):
		img = nib.load(f)
		header = img.header
		data_shape = header.get_data_shape()
		if data_shape != (208, 300, 320):
			print(f)
