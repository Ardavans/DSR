import sys, math
import h5py

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import DPGMM
from sklearn.decomposition import PCA

from collections import defaultdict
from PIL import Image, ImageChops

path_SR = 'Bottleneck4_m.h5'
path_state = 'Bottleneck4_s.h5'

def get_clusters(data, k):
	model = SpectralClustering(n_clusters=k, gamma = 0.3)
	# model = DPGMM(n_components = k)
	return model.fit_predict(data)
	# model.fit(data)
	# return model.row_labels_

def parse_csv(path):
	# csv to numpy array
	data = genfromtxt(path, delimiter=',')
	return data

def parse_hd5(path, name):
	# csv to numpy array
	myFile = h5py.File(path, 'r')
	data = myFile[name]
	data = np.array(data)
	return data

def show_image(full_data, indices, name):
	# try:
	data = []
	for i in indices:
		data.append(full_data[i].reshape(84,84))
	data = np.array(data)
	data = np.mean(data, axis = 0)
	print data.shape
	im = Image.fromarray(255*data)
	# except:
	# 	print 'empty indices'
	im = im.convert('L')
	im.save(name)

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

def map_unique_indices(data, indices):
	unique_map = []
	for i in indices:
		unique_map.append(data[i])
	return np.array(unique_map)

def get_clustered_states(data_set, k, cluster_labels):
	clustered_states = defaultdict(list)
	for i,label in enumerate(cluster_labels):
		# clustered_states[label].append(data_set[i])
		clustered_states[label].append(i)
	return clustered_states

def show_cluster(clustered_states):

	fin_image = Image.fromarray(255*clustered_states[0].reshape(84,84))
	fin_image = fin_image.convert('L')
	for i in range(1,len(clustered_states)):
		cur_image = Image.fromarray(255*clustered_states[i].reshape(84,84))
		cur_image = cur_image.convert('L')
		fin_image = ImageChops.darker(fin_image, cur_image)

	fin_image.show()


if __name__ == '__main__':
	print '...loading states'
	data_s = parse_hd5(path_state, 's_full_tensor')[:10000]

	print '...loading successors'
	data_m2 = parse_hd5(path_SR, 'm_full_tensor')[:10000]

	print '...getting clusters'
	clusters = get_clusters(data_m2, 4)

	n_clusters = len(set(clusters))
	print n_clusters

	clustered_states = get_clustered_states(data_s, n_clusters, clusters)

	for label,cluster in clustered_states.items():
		show_image(data_s, cluster, str(label)+'.bmp')

