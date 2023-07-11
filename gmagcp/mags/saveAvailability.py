import numpy as np
import groundmag as gm
import PyFileIO as pf
from ..Tools.checkPath import checkPath
from .. import globs

def saveAvailability(estn,pstn):
	'''
	Save the availability for a pair of magnetometers
	
	'''
	edates = gm.GetDataAvailability(estn)
	pdates = gm.GetDataAvailability(pstn)


	dates = []
	for i in range(0,edates.size):
		for j in range(0,pdates.size):
			if edates[i] == pdates[j]:
				dates.append(edates[i])
				
	dates = np.array(dates)
	
	fpath = globs.dataPath + '/availability'
	checkPath(fpath)
	
	fname = fpath + '/{:s}-{:s}.bin'.format(estn,pstn)
	
	f = open(fname,'wb')
	pf.ArrayToFile(dates,'int32',f)
	f.close()
