import cPickle

f = file('rocValsSDA','rb')

data = cPickle.load(f)

output = file('rocValsSDAFormatted','wb')

output.write('class\tscore\n')

for i in xrange(0,1600):
	classval = data[i][1][0]
	if classval == 0:
		classstring = '-1'
	else:
		classstring = '+1'
	
	output.write(classstring + '\t%f\n' % (data[i][0][0][0]))
output.close()
