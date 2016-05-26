import zmq
import sys
import numpy as np
from numpy import genfromtxt

from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster import KMeans

port = "1728"
name = sys.argv[1]
tmp_path_in = './temp/' + name + '.csv'
print tmp_path_in
# tmp_path_in = '/om/user/ardavans/DeepSR/dsr/m2_max.csv'
# if len(sys.argv) > 1:
# 	port = int(sys.argv[1])

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)


def get_clusters(data, k):
	# model = SpectralCoclustering(n_clusters=4, random_state=0)
	# model.fit(data)
	# return model.row_labels_
	model = KMeans(n_clusters=3)
	return model.fit_predict(data)

def parse():
	# converts csv file written by lua to a numpy array
	data = genfromtxt(tmp_path_in, delimiter=',')
	return data

while True:

	message = socket.recv()

	data = parse()
	clusters = get_clusters(data, 4)
	return_string = 'clusters = {'

	for i in clusters:
		return_string += str(i) + ','
	clusters = clusters[:-1]

	return_string += '}'

	socket.send(return_string)
