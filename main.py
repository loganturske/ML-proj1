# Load CSV
import numpy

def winnow_algo(filename, alpha, theta):

	raw_data = open(filename, 'rt')
	data = numpy.loadtxt(raw_data, delimiter=",")
	# print(data.shape)
	# print data[1]

	# Randomize data
	numpy.random.shuffle(data)
	# print data[1]

	# Make test and training sets
	split = data.shape[0] / 3
	test_set, training_set = data[:split,:], data[split:, :]
	# print training_set.shape
	# print test_set.shape

	weight_set = numpy.ones(data.shape[1]-1)

	for row in training_set:
		f_of_x = 0
		h_of_x = 0 
		weight_to_use = 0
		for w in row[1:]:
			f_of_x += w * weight_set[weight_to_use]
			weight_to_use += 1

		if f_of_x > theta:
			h_of_x = 1
		else:
			h_of_x = 0

		if h_of_x != row[0]:
			if h_of_x == 0 and row[0] == 1:
				# print "Pro"
				# # Promotion
				weight_to_change = 0
				for x in row[1:]:
					if x:
						weight_set[weight_to_change] = weight_set[weight_to_change] * alpha
					weight_to_change += 1

			if h_of_x == 1 and row[0] == 0:
				# print "Demo"
				# Demotion
				weight_to_change = 0
				for x in row[1:]:
					if x:
						weight_set[weight_to_change] = weight_set[weight_to_change] / alpha
					weight_to_change += 1
	print "Weights:"
	print weight_set

	num_correct = 0
	for row in test_set:
		f_of_x = 0
		h_of_x = 0 
		weight_to_use = 0
		for w in row[1:]:
			f_of_x += w * weight_set[weight_to_use]
			weight_to_use += 1

		if f_of_x > theta:
			h_of_x = 1
		else:
			h_of_x = 0

		# print "Guess: " + str(h_of_x) + " Correct: " + str(row[0])
		if float(h_of_x) == float(row[0]):
			num_correct += 1

	print "Results:"
	print float(float(num_correct) / float(test_set.shape[0]))

def main():
	# winnow_algo('datasets/test_dataset.data', 2, .5)
	winnow_algo('datasets/breast-cancer-wisconsin.data', 2, .5)

if __name__ == "__main__":
    main()