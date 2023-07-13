import numpy as np
import groundmag as gm
import PyFileIO as pf
from ..tools.checkPath import checkPath
from .. import globs
from .listMagPairs import listMagPairs

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


def saveNetworkAvailability(Network=None,dLat=5.0,dLon=2.5):

	pairs = listMagPairs(Network=Network,dLat=dLat,dLon=dLon)

	for i,p in enumerate(pairs):
		print('Saving pair availability {:d} of {:d} ({:s}-{:s})'.format(i+1,len(pairs),p[0],p[1]))
		saveAvailability(p[0],p[1])