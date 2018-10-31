import csv
import random
import math
import operator
import numpy as np

def load_dataset(filename, split, training_set=[] , test_set=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            training_set.append(dataset[x])
	        else:
	            test_set.append(dataset[x])

def calculate_euclidean_distance(instance1, instance2, length):
	distance_between_points = 0
	for x in range(length):
		distance_between_points += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance_between_points)


def fetch_neighbors(training_set, test_instance, k):
	distances = []
	length = len(test_instance)-1
	for x in range(len(training_set)):
		distance_between_points = calculate_euclidean_distance(test_instance, training_set[x], length)
		distances.append((training_set[x], distance_between_points))
	distances.sort(key=operator.itemgetter(1))
	neighbors_list = []
	for x in range(k):
		neighbors_list.append(distances[x][0])
	return neighbors_list


def fetch_response(neighbors_list):
	class_votes = {}
	for x in range(len(neighbors_list)):
		response = neighbors_list[x][-1]
		if response in class_votes:
			class_votes[response] += 1
		else:
			class_votes[response] = 1
	sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sorted_votes[0][0]

def knn(training_set, test_set):
	knn_predictions=[]
	k = 3
	for x in range(len(test_set)):
		neighbors_list = fetch_neighbors(training_set, test_set[x], k)
		result = fetch_response(neighbors_list)
		knn_predictions.append(result)
	print(knn_predictions)

training_set=[]
test_set=[]
split = 0.67
load_dataset('iris.data', split, training_set, test_set)
knn(training_set, test_set)