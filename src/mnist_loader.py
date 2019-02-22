#!/usr/bin/python

"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import csv

# Third-party libraries
import numpy as np

def load_data():
	"""Return the MNIST data as a tuple containing the training data,
	the validation data, and the test data.

	The ``training_data`` is returned as a tuple with two entries.
	The first entry contains the actual training images.  This is a
	numpy ndarray with 50,000 entries.  Each entry is, in turn, a
	numpy ndarray with 784 values, representing the 28 * 28 = 784
	pixels in a single MNIST image.

	The second entry in the ``training_data`` tuple is a numpy ndarray
	containing 50,000 entries.  Those entries are just the digit
	values (0...9) for the corresponding images contained in the first
	entry of the tuple.

	The ``validation_data`` and ``test_data`` are similar, except
	each contains only 10,000 images.

	This is a nice data format, but for use in neural networks it's
	helpful to modify the format of the ``training_data`` a little.
	That's done in the wrapper function ``load_data_wrapper()``, see
	below.
	"""
	
	data_file = "../data/gh.csv"  #dukacopy data hourly gold price
	file_pointer = open(data_file, "r")
	price_record_stream = csv.reader(file_pointer)
	training_data = []
	holder = []
	counter = 0
	sub_counter = 0
	price_data = np.zeros(784, dtype='f')
	open_price = 0
	close_price = 0

	for row in price_record_stream:  #create input data structure for neural network
		print "hi"

		open_price  = float(row[1])
		prev_close_price = close_price
		close_price = float(row[2])
		price_data[sub_counter] = open_price / 3000.0
		sub_counter += 1
		counter += 1
		
		if counter % 784 == 0:
			
			holder = (price_data, vectorized_answer(prev_close_price))

			training_data.append(holder)
			sub_counter = 0
			price_data = np.zeros(784)

		if counter >= 47040:
			break
			
	print training_data 
	file_pointer.close()
	
	return
	return (training_data, validation_data, test_data)

def load_data_wrapper():
	"""Return a tuple containing ``(training_data, validation_data,
	test_data)``. Based on ``load_data``, but the format is more
	convenient for use in our implementation of neural networks.

	In particular, ``training_data`` is a list containing 50,000
	2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
	containing the input image.  ``y`` is a 10-dimensional
	numpy.ndarray representing the unit vector corresponding to the
	correct digit for ``x``.

	``validation_data`` and ``test_data`` are lists containing 10,000
	2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
	numpy.ndarry containing the input image, and ``y`` is the
	corresponding classification, i.e., the digit values (integers)
	corresponding to ``x``.

	Obviously, this means we're using slightly different formats for
	the training data and the validation / test data.  These formats
	turn out to be the most convenient for use in our neural network
	code."""
	tr_d, va_d, te_d = load_data()
	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_results = [vectorized_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, te_d[1])
	return (training_data, validation_data, test_data)

def vectorized_result(j):
	"""Return a 10-dimensional unit vector with a 1.0 in the jth
	position and zeroes elsewhere.  This is used to convert a digit
	(0...9) into a corresponding desired output from the neural
	network."""
	e = np.zeros((10, 1))
	e[j] = 1.0
	return e
	
def vectorized_answer(close_price):
	close_price = long(round(close_price * 100, 0))
	e = np.zeros((17, 1))
	for i in range(17):
		bn = "{0:017b}".format(close_price)
		e[i] = bn[i]
	print e
	return e
	
load_data()

