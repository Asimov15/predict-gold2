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

image_size           	= 784    # in this case how many hours back the neural net will study to determinethe next price movement
training_data_size   	= 60  # the number of set of the above to train with
validation_data_size 	= 10
test_data_size       	= 10

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
	
	data_file 				= "../data/gh.csv"  #dukascopy data hourly gold price csv
	file_pointer 			= open(data_file, "r") #create a file pointer to the csv file
	price_record_stream 	= csv.reader(file_pointer) # create the record stream
	training_data 			= [] # create a np array to hold the training data
	validation_data 		= [] # create a list to hold the validation data
	test_data       		= []
	holder 					= np.zeros((2))
	holder_val 				= np.zeros((2))
	holder_test 			= np.zeros((2))

	counter 				= 0
	sub_counter 			= 0

	price_rec   			= np.ndarray((image_size),  		 dtype='float32')
	train_answer   			= np.ndarray((training_data_size),	 dtype='float32')
	val_answer   			= np.ndarray((validation_data_size), dtype='float32')
	test_answer   			= np.ndarray((test_data_size),		 dtype='float32')
	
	#train_price_data		= np.ndarray((training_data_size),	 dtype='ndarray')
	
	train_price_data		= []
	val_price_data			= []
	test_price_data			= []
	
	open_price 				= 0
	close_price 			= 0
	prev_close_price 		= 0
	train_image_counter 	= 0
	val_image_counter		= 0
	test_image_counter		= 0
	new_limit_val 			= 0
	new_limit_test 			= 0

	for row in price_record_stream:  #create input data structure for neural network
		open_price  = float(row[1])
		prev_close_price = close_price
		close_price = float(row[2])
		price_rec[sub_counter] = float(open_price / 3000.0)
		sub_counter += 1
		counter += 1
		new_limit_val  = training_data_size + validation_data_size
		new_limit_test = training_data_size + validation_data_size + test_data_size

		if counter % image_size == 0:
			if counter <= image_size * training_data_size:
				train_price_data.append(price_rec)
				#train_price_data[counter/image - 1] = price_rec 
				train_answer[train_image_counter] = prev_close_price
				train_image_counter += 1
				training_data = (train_price_data, train_answer)
			elif counter > image_size * training_data_size and counter <= image_size * new_limit_val:
				val_price_data.append(price_rec)
				val_answer[val_image_counter] = prev_close_price
				val_image_counter += 1
				validation_data = (val_price_data, val_answer)
			elif counter > image_size * new_limit_val and counter <= image_size * new_limit_test:
				test_price_data.append(price_rec)
				test_answer[test_image_counter] = prev_close_price
				test_image_counter += 1
				test_data = (test_price_data, test_answer)
			else:
				break

			sub_counter = 0
			price_rec   = np.ndarray((image_size),  dtype='float32')

	file_pointer.close()

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

	training_inputs = [np.reshape(x, (image_size, 1)) for x in tr_d[0]]
	training_results = tr_d[1]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (image_size, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (image_size, 1)) for x in te_d[0]]
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
	return e
	
load_data_wrapper()

