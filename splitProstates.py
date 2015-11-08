import numpy
import cPickle
import random

f = file('pickledProstates', 'rb')
lf = file('ProstateDataLeader.pkl', 'rb');
covariates, targets = cPickle.load(f)
l_covariates = cPickle.load(lf);


mins = numpy.ndarray((131,),float);
mins.fill(99999)
maxes = numpy.ndarray((131,),float);
maxes.fill(-99999)

for i in xrange(0,covariates.shape[0]):
	for j in xrange(0, 131):
		mins[j] = min(mins[j], covariates[i][j])
		maxes[j] = max(maxes[j], covariates[i][j])

for i in xrange(0,l_covariates.shape[0]):
	for j in xrange(0,131):
		mins[j] = min(mins[j], l_covariates[i][j])
		maxes[j] = max(maxes[j], l_covariates[i][j])



for i in xrange(0,covariates.shape[0]):
	for j in xrange(0,131):
		if(mins[j] == maxes[j]):
			if(covariates[i][j] != 0):
				covariates[i][j] = covariates[i][j] / covariates[i][j]
		else:
			covariates[i][j] = (covariates[i][j] - mins[j]) / (abs(maxes[j] - mins[j]))


for i in xrange(0,l_covariates.shape[0]):
	for j in xrange(0,131):
		if(mins[j] == maxes[j]):
			if l_covariates[i][j] != 0:
				l_covariates[i][j] = l_covariates[i][j] / l_covariates[i][j]
		else:
			l_covariates[i][j] = (l_covariates[i][j] - mins[j]) / (abs(maxes[j] - mins[j]))		




f.close()
lf.close()

ndf = file('pickledProstatesNormalized.pkl','wb')
nldf = file('leaderProstateDataNormalized.pkl','wb')
cPickle.dump((covariates,targets),ndf,protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(l_covariates, nldf, protocol=cPickle.HIGHEST_PROTOCOL)
ndf.close()
nldf.close()



