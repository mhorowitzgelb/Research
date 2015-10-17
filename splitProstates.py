import numpy
import cPickle
import random

f = file('pickledProstates', 'rb')

covariates, targets = cPickle.load(f)


mins = numpy.ndarray((131,),float);
mins.fill(99999)
maxes = numpy.ndarray((131,),float);
maxes.fill(-99999)

for i in xrange(0,covariates.shape[0]):
	for j in xrange(0, 131):
		mins[j] = min(mins[j], covariates[i][j])
		maxes[j] = max(maxes[j], covariates[i][j])

for i in xrange(0,covariates.shape[0]):
	for j in xrange(0,131):
		if(mins[j] == maxes[j]):
			if(covariates[i][j] != 0):
				covariates[i][j] = covariates[i][j] / covariates[i][j]
		else:
			covariates[i][j] = (covariates[i][j] - mins[j]) / (abs(maxes[j] - mins[j]))






f.close()




covariates_validation = numpy.ndarray((160,131))
targets_validation = numpy.ndarray((160,))

covariates_testing = numpy.ndarray((160,131))
targets_testing = numpy.ndarray((160,))

for i in xrange(0,160):
	index = random.randint(0, covariates.shape[0]-1)
	covariates_validation[i] = covariates[index]
	targets_validation[i] = targets[index]
	covariates = numpy.delete(covariates, index, axis=0)
	targets = numpy.delete(targets, index, axis=0)
	

	index = random.randint(0, covariates.shape[0]-1)
	covariates_testing[i] = covariates[index]
	targets_testing[i] = targets[index]
	covariates = numpy.delete(covariates, index, axis=0)
	targets = numpy.delete(targets, index, axis=0) 

covariates_training = covariates
targets_training = targets

dataset = [(covariates_training,targets_training), (covariates_validation,targets_validation), (covariates_testing,targets_testing)]


f = file('pickledProstatesDivided', 'wb')

cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()





