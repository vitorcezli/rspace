#!/usr/bin/python3
from __future__ import division
import sklearn
import numpy as np
import math
import csv

# data must be a matrix
# indexes_r is a list that specifies the subspace to be removed
# delta_r is a list that specifies the delta value on each variable
# c_list is a classification list
# d_or_c is a list that specifies for each variable if it is discrete or continuous
def get_error(data, indexes_r, delta, c_list, d_or_c):
	"Returns the space error if the specified subspace is removed"
	# initializes the necessary variables to execute the algorithm
	classifications = list(set(c_list))
	max_list = []
	min_list = []
	discrete_list = []
	dict_space = {}

	# put in the variables necessary values
	for column in range(len(data[0])):
		column = int(column)
		if d_or_c[column]: # variable is discrete
			max_list.append(0)
			min_list.append(0)
			discrete_list.append(list(set([i[column] for i in data])))
		else:
			max_list.append(max([i[column] for i in data]))
			min_list.append(min([i[column] for i in data]))
			discrete_list.append([])
	# saves the classification of each point
	for i in range(len(data)):
		hash_indexes = [] # it will indicate where the point will be saved
		for j in range(len(data[i])):
			if j not in indexes_r:
				if d_or_c[j]:
					hash_indexes.append(discrete_list[j].index(data[i][j]))
				elif isinstance(delta, list):
					hash_indexes.append(math.floor((data[i][j] - min_list[j]) / delta[j]))
				else:
					hash_indexes.append(math.floor((data[i][j] - min_list[j]) / delta))
		dict_space[tuple(hash_indexes)] = \
			dict_space.get(tuple(hash_indexes), []) + [c_list[i]]

	# calculates statistical values
	mean = np.mean([len(i) for i in list(dict_space.values())])
	des = math.sqrt(np.var([len(i) for i in list(dict_space.values())]))

	# calculates R error
	error = 0
	values = list(dict_space.values())
	total = 0
	prob_matrix = []
	for line in values:
		most_element = most_common(line)
		length = len(line)
		prob_matrix.append([line.count(most_element) / length, length])
		total += length
	for line in prob_matrix:
		error += (line[0] * line[1] / total)
	return [error, mean, des]