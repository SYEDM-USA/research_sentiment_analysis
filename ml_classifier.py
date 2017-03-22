# K-Nearest Neighour Classifier
import numpy as np


def distance(_object_1, _object_2):
	# Computes squared euclidean distance
	return np.sum( (_object_1 - _object_2) ** 2  )

def Knn_classifier(_training_set, _training_label, _new_example):
	dists = np.array([distance(t, _new_example) for t in _training_set])
	nearst = dists.argmin()
	return _training_label[nearst];

