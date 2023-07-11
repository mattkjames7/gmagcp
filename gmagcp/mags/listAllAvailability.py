import numpy as np
from ..tools.listFiles import listFiles
from .readAvailability import readAvailability
import os
from .. import globs


def listAllAvailability(dateLim=[19950101,20041231]):
	
	fpath = globs.dataPath + '/Availability'
	
	
	_,fnames = listFiles(fpath,ReturnNames=True)
	
	nf = fnames.size
	
	out = []
	for i in range(0,nf):
		nm,ex = os.path.splitext(fnames[i])
		estn,pstn = nm.split('-')
		
		dates = readAvailability(estn,pstn)
		
		for d in dates:
			if dateLim is None:
				out.append((estn,pstn,d))
			elif ((d >= dateLim[0]) & (d <= dateLim[1])):
				out.append((estn,pstn,d))


	return out
