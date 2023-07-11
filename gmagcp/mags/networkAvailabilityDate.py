import numpy as np
from .listMagPairs import listMagPairs
from .readAvailability import readAvailability

def networkAvailabilityDate(Date,Network):
	

	pairs = listMagPairs(Network)
	
	n = len(pairs)
	avail = []
	estn = []
	pstn = []
	for p in pairs:
		estn.append(p[0])
		pstn.append(p[1])
	
		dates = readAvailability(p[0],p[1])
		avail.append(Date in dates)
		
	return np.array(estn),np.array(pstn),np.array(avail)
