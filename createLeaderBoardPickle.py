import csv
import numpy
import cPickle


array = numpy.zeros(shape = (157,131), dtype=float) 


with open('ProstateDataLeader.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	i = -1
	for row in reader:
		if i >= 0:
			array[i] = row[1:-1]
		i += 1	
	
f = file('ProstateDataLeader.pkl', 'wb')

cPickle.dump(array, f, cPickle.HIGHEST_PROTOCOL)
