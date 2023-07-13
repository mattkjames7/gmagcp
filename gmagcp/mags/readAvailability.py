import numpy as np
import PyFileIO as pf
from .. import globs
import os

def readAvailability(estn,pstn):
	
	
	fpath = globs.DataPath + '/Availability'	
	fname = fpath + '/{:s}-{:s}.bin'.format(estn,pstn)
	
	if not os.path.isfile(fname):
		return np.array([],dtype='int32')
	
	f = open(fname,'rb')
	dates = pf.ArrayFromFile('int32',f)
	f.close()
	
	return dates
