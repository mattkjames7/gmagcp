import numpy as np
import groundmag as gm

def listMagPairs(Network=None,dLat=5.0,dLon=2.5):
	
	stns = gm.GetStationInfo()
	
	if not Network is None:
		gNet = np.zeros(stns.size,dtype='bool')
		for i in range(0,stns.size):
			if stns.Network[i] in Network:
				gNet[i] = True
		gNet = np.array([gNet])
		gNet = gNet & gNet.T		
	else:
		gNet = np.ones((stns.size,stns.size),dtype='bool8')
	
	
	code = np.array([stns.Code])
	gMag = code != code.T
	
	
	lon = stns.mlon
	lat = stns.mlat
		
	dlon = np.abs(np.array([lon]) - np.array([lon]).T) <= dLon
	dlat = np.abs(np.array([lat]) - np.array([lat]).T) <= dLat
	
	I = np.bool8(1 - np.identity(stns.size))
	
	I = np.triu(I).astype('bool8')
	
	use0,use1 = np.where(dlon & dlat & I & gNet & gMag)


	pairs = []
	for i in range(0,use0.size):
		if stns.mlat[use0[i]] < stns.mlat[use1[i]]:
			pairs.append((stns.Code[use0[i]],stns.Code[use1[i]]))
		else:
			pairs.append((stns.Code[use1[i]],stns.Code[use0[i]]))
			
	return pairs
