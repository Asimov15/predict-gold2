#!/usr/bin/python

import numpy as np

t = np.ndarray((10), dtype='void')
a = np.ndarray((10), dtype='float') 

for b in range(10):
    a[b] = b

t[0] = a
