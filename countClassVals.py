import numpy, gzip, cPickle, theano

sum = 0

with gzip.open('pickledProstatesDivided.pkl.gz','rb') as f:
	data =  cPickle.load(f)
	for i in xrange(0,160):
		sum += data[1][1][i]

print sum
