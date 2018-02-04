# Load CSV
import numpy

filename = 'datasets/breast-cancer-wisconsin.data'
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
print training_set.shape
print test_set.shape
