#!/usr/bin/python


#"""
#network.py
#~~~~~~~~~~

#A module to implement the stochastic gradient descent learning
#algorithm for a feedforward neural network.  Gradients are calculated
#using backpropagation.  Note that I have focused on making the code
#simple, easily readable, and easily modifiable.  It is not optimized,
#and omits many desirable features.

#mnist_loader
#~~~~~~~~~~~~

#A library to load the MNIST image data.  For details of the data
#structures that are returned, see the doc strings for ``load_data``
#and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
#function usually called by our neural network code.
#"""

#### Libraries
# Standard library
import csv
import random

# Third-party libraries
import numpy as np

image_size		   	= 784	# in this case how many hours back the neural net will study to determinethe next price movement
training_data_size   	= 1000  # the number of set of the above to train with
validation_data_size 	= 100
test_data_size	   	= 100

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
	
	data_file 				= "../data/gm3.csv"  #dukascopy data hourly gold price csv
	file_pointer 			= open(data_file, "r") #create a file pointer to the csv file
	price_record_stream 	= csv.reader(file_pointer) # create the record stream
	training_data 			= [] # create a np array to hold the training data
	validation_data 		= [] # create a list to hold the validation data
	test_data	   		= []
	holder 					= np.zeros((2))
	holder_val 				= np.zeros((2))
	holder_test 			= np.zeros((2))

	counter 				= 0
	sub_counter 			= 0

	price_rec   			= np.ndarray((image_size),  		 dtype='float64')
	train_answer   			= np.ndarray((training_data_size),	 dtype='float64')
	val_answer   			= np.ndarray((validation_data_size), dtype='float64')
	test_answer   			= np.ndarray((test_data_size),		 dtype='float64')
	
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
	training_results = [vectorized_answer(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (image_size, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (image_size, 1)) for x in te_d[0]]
	training_results = [round(100 * y) for y in te_d[1]]
	test_data = zip(test_inputs, training_results)

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
	e = np.zeros((18, 1))
	for i in range(18):
		bn = "{0:018b}".format(close_price)
		e[i] = bn[i]
	return e


class Network(object):

	def __init__(self, sizes):
		"""The list ``sizes`` contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta,
			test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(
					j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					   for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			sigmoid_prime(zs[-1])
		#print delta
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in xrange(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		#print test_data
		test_results = []
		for (x, y) in test_data:
			b = self.feedforward(x)
			print b
			print "--"
			a = self.calc_price(b)
			print a
			test_results.append((a,y))
		#test_results = [(self.calc_price(self.feedforward(x)), y) for (x, y) in test_data]

		#for (x, y) in test_results:
			#print x, y
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return (output_activations-y)
	
	def calc_price(self, v):
		x = 0
		for i in range(18):
			x += round(v[i]) * pow(2, 18-i)
			#print x
		return x

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))

print "hello"
training_data, validation_data, test_data = load_data_wrapper()
print "hello2"
net = Network([784, 3000, 18])
print "hello3"

#print training_data

#print test_data
print len(test_data)
net.SGD(training_data, 500, 80, 20.0, test_data=test_data)
