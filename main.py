# Load CSV
import numpy
import math

def winnow_algo(test_set, training_set, alpha, theta):

	# Set a weight set to be all ones
	weight_set = numpy.ones(test_set.shape[1]-1)

	# For every instance of the training set
	for row in training_set:

		f_of_x = 0
		h_of_x = 0 
		weight_to_use = 0

		# For each attribute (other than the answer) multiply by the weight corresponding
		#	to the weight set
		for w in row[1:]:
			f_of_x += w * weight_set[weight_to_use]
			weight_to_use += 1

		# If your f(x) is more than the theta threshold, classiy as a 1, otherwise, 0
		if f_of_x > theta:
			h_of_x = 1
		else:
			h_of_x = 0

		# If you guessed wrong
		if h_of_x != row[0]:
			if h_of_x == 0 and row[0] == 1:
				# Promotion
				weight_to_change = 0
				# For every attribute that was a 1, multiply by alpha
				for x in row[1:]:
					if x:
						weight_set[weight_to_change] = weight_set[weight_to_change] * alpha
					weight_to_change += 1

			if h_of_x == 1 and row[0] == 0:
				# Demotion
				weight_to_change = 0
				# For every attribute that was a 1, multiply by alpha
				for x in row[1:]:
					if x:
						weight_set[weight_to_change] = weight_set[weight_to_change] / alpha
					weight_to_change += 1
	print "Winnow-2 weight model:"
	print weight_set

	num_correct = 0
	# For every instance in the test set
	for row in test_set:
		f_of_x = 0
		h_of_x = 0 
		weight_to_use = 0

		# Preform the algo like before without updating weight set
		for w in row[1:]:
			f_of_x += w * weight_set[weight_to_use]
			weight_to_use += 1

		if f_of_x > theta:
			h_of_x = 1
		else:
			h_of_x = 0

		# If you guessed right then increase the number of correct guesses
		if float(h_of_x) == float(row[0]):
			num_correct += 1

	accuracy = float(float(num_correct) / float(test_set.shape[0]))
	print('Winnow Accuracy: {0}%').format(accuracy*100)

def load_CSV(filename):
	# Open the file to read
	raw_data = open(filename, 'rt')
	# Create a numpy list on the comma delimiter
	data = numpy.loadtxt(raw_data, delimiter=",")
	# For all of the rows in the list, change to floats
	for i in range(len(data)):
		data[i] = [float(x) for x in data[i]]
	return data

def split_data(data):
	# Randomize data
	numpy.random.shuffle(data)
	
	# Make test and training sets
	split = data.shape[0] / 3
	test_set, training_set = data[:split,:], data[split:, :]
	return test_set, training_set

def separate_by_class(data):
	# Create an empty set for put the classes, for now only going to be 1 or 0
	classes = {}
	# For every row in the data, get all the classes and put them into the 'classes' set
	for i in range(len(data)):
		row = data[i]
		if (row[0] not in classes):
			classes[row[0]] = []
		classes[row[0]].append(row)
	return classes

def mean(arr):
	# Get the mean of the entire array of numbers that was passed in
	return sum(arr)/float(len(arr))

def stdev(arr):
	# Get the standard deviation of the entire array that was passed in
	# Start by getting the variance
	variance = sum([pow(x-mean(arr),2) for x in arr])/float(len(arr)-1)
	print variance
	# Then take the square root of the variance
	return math.sqrt(variance)

def summarize(data):
	# the "zip" function will give us an iterable for each row of the data so we can get the attributes
	#	for evey instance of data
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*data)]
	# Remove the summaries for the first attribute which is the "answer"
	del summaries[0]
	return summaries

def summarize_by_class(data):
	# Separate all of the classes
	separated = separate_by_class(data)
	summaries = {}
	# Get the summaries for each of the separated classes
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculate_probability(x, mean, stdev):
	if (2*math.pow(stdev,2)) == 0:
		return 1
	# Gaussian Probablility
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, inputVector):
	# Get the probabilities of all of the summaries
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		# For each class summary calculate the probability and multiply it with the others
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculate_probability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	# Get the probability of all the classes
	probabilities = calculate_class_probabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	# For each of the probabilies get the the best
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def get_predictions(summaries, testSet):
	# Go through the test set and make predictions
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def get_accuracy(testSet, predictions):
	# Calculate Accuracy
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][0] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def naive_bayes_algo(test_set, training_set):
	# Summarize the attributes
	summaries = summarize_by_class(training_set)
	print summaries
	# Get the predictions of the test set
	predictions = get_predictions(summaries, test_set)
	#Get the accuracy
	accuracy = get_accuracy(test_set, predictions)
	print('Naive Bayes Accuracy: {0}%').format(accuracy)

def main():

	print "############ breast-cancer-wisconsin.data ############"
	data = load_CSV('datasets/breast-cancer-wisconsin.data')
	test_set, training_set = split_data(data)
	winnow_algo(test_set, training_set, 2, .5)
	naive_bayes_algo(test_set, training_set)

	print "############ house-votes-84.data ############"
	data = load_CSV('datasets/house-votes-84.data')
	test_set, training_set = split_data(data)
	winnow_algo(test_set, training_set, 2, .5)
	naive_bayes_algo(test_set, training_set)

	print "############ iris.data ############"
	data = load_CSV('datasets/iris.data')
	test_set, training_set = split_data(data)
	winnow_algo(test_set, training_set, 2, .5)
	naive_bayes_algo(test_set, training_set)

if __name__ == "__main__":
    main()