#!/usr/bin/python
# Author: Jeremiah Clothier
# CIS 472/572 Machine Learning (Spring 2019) -- Programming Homework #1
# Starter code provided by Daniel Lowd
#
import sys
import re
import math
# Node class for the decision tree
import node


train=None
varnames=None
test=None
testvarnames=None
root=None


def entropy(p):
	"""
	Helper function computes entropy of Bernoulli distribution with
	parameter p
	"""
	# if the probability is 1 or 0 then the value is 0
	if p == 1 or p == 0:
		return 0
	# Otherwise calculate using the entropy formula for binary classification
	return (-p *math.log(p,2)) - ((1-p) * math.log((1-p),2))
	

def collect_counts(data):
	"""
	Collects and returns the following information:
		- list[int]: collect counts for each variable value with each class label (where the classification is positive)
		- list[int]: collect counts for each variable value with each class label
		- int: totals the number of positive classifications
		- int: totals the number of data entries
	"""
	length = class_index = len(data[0])-1 # subtract 1, otherwise index error
	
	columns_count = [sum(column) for column in zip(*data)] # columns collapsed to one item via sum
	positive_classifications_count, columns_count = columns_count[-1], columns_count[:-1] # cut off the last element because that is the class count
	
	# collect counts for each variable value with each class label (where the classifcation is positive)
	positive_colums_count = [0 for _ in range(length)]
	for entry in data:
		if entry[class_index] == 1:
			for i in range(length):
				if entry[i] == 1:
					positive_colums_count[i] += 1
  	return (positive_colums_count, columns_count, positive_classifications_count, len(data))


def infogain(positive_column_count, column_count, positive_classifications_count, ttl):
	"""
	Compute information gain for a particular split, given the counts
	positive_column_count : number of occurences of y=1 with x_i=1 for all i=1 to n
	column_count : number of occurrences of x_i=1
	positive_classifications_count : number of ocurrences of y=1
	ttl : total length of the data
	"""
	# Avoid division by zero error
	if column_count == 0 or column_count == ttl:
		return 0
	
	# python2.7 division of two ints yields an integer, therefore cast variables to a floats
	positive_column_count = float(positive_column_count)
	column_count = float(column_count)
	positive_classifications_count = float(positive_classifications_count)
	ttl = float(ttl)
	
	# calculate proportions and then perform arithmetic for gain formula on a binary classifcation problem
	positive_column_count_prob = positive_column_count / column_count

	# number of occurrences of y = 1 where x_i = 0/ Number of occurrences of x_i = 0
	negative_column_count_prob = (positive_classifications_count - positive_column_count) / (ttl - column_count)
	column_count_prob = column_count/ttl
	
	# Return the evaluation of the gain formula for a binary classifciation problem
	return entropy(positive_classifications_count/ttl) \
		- (column_count_prob) * entropy(positive_column_count_prob) \
			- (1-column_count_prob) * entropy(negative_column_count_prob)


def maxgain(data):
	"""
	returns the candidate index variable (corresponding to the index which gave us the maxgain across all candidate variables)
	"""
	positive_colums_count, columns_count, positive_classifications_count, ttl = collect_counts(data)
	max_val, candidate_index = 0 , -1 # if candidate_index remains negative one then it's a null leaf because the dataset is empty
	for i in range(len(columns_count)):
		gain = infogain(positive_colums_count[i], columns_count[i], positive_classifications_count, ttl)
		if gain > max_val:
			max_val, candidate_index = gain, i
	return candidate_index

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
		data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
	best_candidate = maxgain(data)
	# Base Case: if all our data has the same classification then we can create a leaf node to classify our root->leaf path 
	if best_candidate == -1:
		for entry in data:
			if entry[-1] == 1:
				return node.Leaf(varnames,1)
		return node.Leaf(varnames,0)
	# Recursive Step: split our data into two lists: (1) positive classification (2) negative classifcation. Then
	# build the left and right subtree by calling the 'build_tree' function again.
	else:
		positive_data = []
		negative_data = []
		for entry in data:
			if entry[best_candidate] == 1:
				positive_data.append(entry)
			else:
				negative_data.append(entry)
		# Trees are by defintion recursive, so after splitting data we can continue with the process of building our tree.
		# autograder checks left -> zero and right -> one
		left_node = build_tree(negative_data, varnames)
		right_node = build_tree(positive_data, varnames)
		return node.Split(varnames, best_candidate, left_node, right_node)
		

# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS

	# build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
	root = build_tree(train, varnames)
	print_model(root, modelfile)

def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
        # This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
		print 'Usage: id3.py <train> <test> <model>'
		sys.exit(2)
    loadAndTrain(argv[0],argv[1],argv[2])

    acc = runTest()
    print "Accuracy: ",acc

if __name__ == "__main__":
    main(sys.argv[1:])
