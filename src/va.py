#!/usr/bin/python

import numpy as np

def vectorized_answer(close_price):
	close_price = long(round(close_price * 100, 0))
	e = np.zeros((17, 1))
	for i in range(17):
		bn = "{0:017b}".format(close_price)
		e[i] = bn[i]
	print e

print "hi"

vectorized_answer(1254.98)
exit()
