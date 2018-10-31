import math
import pandas as pd
import numpy as np
import operator

def euclideanDistance(data1, data2, length):
	distance = 0
	for x in range(length):
		distance += np.square(data1[x] - data2[x])
	return np.sqrt(distance)

def nn(trainingSet, testInstance):
	distances = {}
	sort = {}
 
	length = testInstance.shape[1]
	
	for x in range(len(trainingSet)):
		
		dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

		distances[x] = dist[0]
 
	sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
 
	neighbor = sorted_d[0][0]
	response = trainingSet.iloc[neighbor][-1]
	
	return (response, neighbor)

data = pd.read_csv("iris.csv")
testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

result_nn,_ = nn(data, test)

print("NN")
print(result_nn)
