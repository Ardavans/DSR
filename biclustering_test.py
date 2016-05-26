import zmq
import sys
import numpy as np
from numpy import genfromtxt
import h5py
import math

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import DPGMM
from sklearn.decomposition import PCA

from PIL import Image, ImageChops

# name = sys.argv[1]
# name_m2 = 'm2_dsr_DSR_DQN_Bottleneck_seed_1729_game_Bottleneck2Rooms'
# name_s = 's_dsr_DSR_DQN_Bottleneck_seed_1729_game_Bottleneck2Rooms'
name_m2 = 'Bottleneck4_m' 
# name_s = 's_Bottleneck2Rooms'
name_s = 'Bottleneck4_s'
tmp_path_m2 = '/home/cocosci/simanta/DeepSR/' + name_m2 + '.h5'
tmp_path_s = '/home/cocosci/simanta/DeepSR/' + name_s + '.h5'


def get_clusters(data, k):

	# print data.shape
	# pca = PCA(n_components=3)
	# X_red = pca.fit_transform(data)

	# print X_red.shape
	# model = SpectralClustering(n_clusters=k, gamma = 0.1)
	# model = KMeans(n_clusters=k)
	model = DPGMM(n_components = 10)
	return model.fit_predict(data)
	# model.fit(data)
	# return model.row_labels_

def parse(path):
	# converts csv file written by lua to a numpy array
	data = genfromtxt(path, delimiter=',')
	return data

def parse_hd5(path, name):
	myFile = h5py.File(path, 'r')
	data = myFile[name]
	data = np.array(data)
	return data

def show_image(full_data, indices):
	# try:
	data = []
	for i in indices:
		data.append(full_data[i].reshape(84,84))
	data = np.array(data)
	data = np.mean(data, axis = 0)
	print data.shape
	im = Image.fromarray(255*data)
	im.show()
	# except:
	# 	print 'empty indices'

def show_image_chops(full_data, indices):
	try:
		for i in range(0, len(indices)-1, 2):
			data1 = full_data[i].reshape(84,84)
			data2 = full_data[i+1].reshape(84,84)
			im1 = Image.fromarray(255*data1)
			im2 = Image.fromarray(255*data2)
			im_temp = ImageChops.multiply(im1,im2)
			try:
				im = ImageChops.multiply(im,im_temp)
			except:
				print "hello"
				im = im_temp
		im.show()
	except:
		print 'empty indices'

def get_unique_states(data_s):
	unique_index_list = [0]
	for i,s in enumerate(data_s):
		unique = True
		for j in unique_index_list:
			if (s == data_s[j]).all():
				unique = False
		if unique:
			unique_index_list.append(i)
	return unique_index_list


data_s = parse_hd5(tmp_path_s, 's_full_tensor')
data_m2 = parse_hd5(tmp_path_m2, 'm_full_tensor')
print '...getting unique states'
unique_index_list = [i for i in range(19000)]
# print unique_index_list
# print len(unique_index_list)

index_mapping = {}
for i,unique_i in enumerate(unique_index_list):
	index_mapping[i] = unique_i

unique_data_m2 = []
for i in unique_index_list:
	unique_data_m2.append(data_m2[i+1])
unique_data_m2 = np.array(unique_data_m2)
# print '-----------------------------------'
# im = Image.fromarray(255*unique_data_m2)
# im.show()
# print unique_data_m2.shape


# # show_image_chops(data_s, [i for i in range(len(data_s))])

print '...getting clusters'
clusters = get_clusters(unique_data_m2, 4)
print '...got clusters'

zero_indices = []
one_indices = []
two_indices = []
three_indices = []
four_indices = []
five_indices = []
six_indices = []
seven_indices = []


for i, cluster in enumerate(clusters):
	if cluster == 0:
		zero_indices.append(i)
	elif cluster == 1:
		one_indices.append(i)
	elif cluster == 2:
		two_indices.append(i)
	elif cluster == 3:
		three_indices.append(i)
	elif cluster == 4:
		four_indices.append(i)
	elif cluster == 5:
		five_indices.append(i)
	elif cluster == 6:
		six_indices.append(i)
	elif cluster == 7:
		seven_indices.append(i)

unique_zero_indices = []
for i in zero_indices:
	unique_zero_indices.append(index_mapping[i])

unique_one_indices = []
for i in one_indices:
	unique_one_indices.append(index_mapping[i])

unique_two_indices = []
for i in two_indices:
	unique_two_indices.append(index_mapping[i])

unique_three_indices = []
for i in three_indices:
	unique_three_indices.append(index_mapping[i])

unique_four_indices = []
for i in four_indices:
	unique_four_indices.append(index_mapping[i])

unique_five_indices = []
for i in five_indices:
	unique_five_indices.append(index_mapping[i])

unique_six_indices = []
for i in six_indices:
	unique_six_indices.append(index_mapping[i])

unique_seven_indices = []
for i in seven_indices:
	unique_seven_indices.append(index_mapping[i])	

show_image(data_s, unique_zero_indices)
show_image(data_s, unique_one_indices)
show_image(data_s, two_indices)
show_image(data_s, three_indices)
show_image(data_s, four_indices)
show_image(data_s, five_indices)
show_image(data_s, six_indices)
show_image(data_s, seven_indices)

# # return_string = 'clusters = {'

# # for i in clusters:
# # 	return_string += str(i) + ','
# # clusters = clusters[:-1]

# # return_string += '}'
# # print return_string


