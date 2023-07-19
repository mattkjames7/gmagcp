import numpy as np
from scipy.interpolate import interp1d
import DateTimeTools as TT
import wavespec as ws
from .. import profile

def _listGaps(good):
	'''
	List start and end indices of gaps in data
	
	'''
	
	#get differences
	i0 = np.where(good[:-1] | (good[1:] == False))[0] + 1 #start of gap index
	i1 = np.where((good[:-1] == False) | (good[1:]))[0] + 1#end of gap index (actually index of first good data point)

	
	#add start and end if needed
	if good[0] == False:
		i0 = np.append(0,i0)
	if good[-1] == False:
		i1 = np.append(i1,good.size)
		
	return i0,i1
	
	
def _listGoodElements(data):
	
	#check for wildly huge values and non-finite values
	bad = np.zeros(data.size,dtype='bool')
	fields = ['Bx','By','Bz']
	for f in fields:
		bad = bad | (np.isfinite(data[f]) == False) | (np.abs(data[f]) > 100000)
	good = bad == False
	
	return good	

def _fillDataGaps(data,MaxGap=300):
	'''
	Try and fill bad data gaps.
	
	
	'''
	#get good/bad elements
	good = _listGoodElements(data)
	bad = good == False

	
	#list all gaps
	i0,i1 = _listGaps(good)
	
	#interpolate each component
	use = np.where(good)[0]
	t = data.utc - data.utc[0]
	fields = ['Bx','By','Bz']
	for f in fields:
		#get interp function
		Fi = interp1d(t[use],data[f][use])
		
		#interpolate 
		for i in range(0,i0.size):
			if i0[i] > 0 and i1[i] < data.size:
				try:
					ind = np.arange(i0[i],i1[i])
					data[f][ind] = Fi(t[ind])
				except:
					raise Exception
	return data

		
def _resampleData(data0,Date,MaxGap=300.0):
	'''
	Resample data to a custom resolution
	
	'''

	
	cfg = profile.get()
	window = cfg['window']
	res = cfg.get('res',1.0)


	
	#get the newtime axis
	n = np.int32(np.round((86400 + 2*window)/res))
	tsec = np.arange(n,dtype='float64')*res - window
	utc0 = TT.ContUT(Date,0.0)[0]
	utc = utc0 + tsec/3600.0
	tsec0 = (data0.utc - utc0)*3600.0	

	#create output recarray
	data = np.recarray(utc.size,dtype=data0.dtype)
	data.Date,data.ut = TT.ContUTtoDate(utc)
	data.utc = utc

	#get gaps in current array
	good = _listGoodElements(data0)
	bad = good == False
	use = np.where(good)[0]
	i0,i1 = _listGaps(good)
	

	#interpolate to new array
	fields = ['Bx','By','Bz']
	for f in fields:
		#create interpolation object
		Fi = interp1d(tsec0[use],data0[f][use],fill_value=np.nan,bounds_error=False)
		
		#interpolate to new array
		data[f] = Fi(tsec)
		
		#fill big gaps with nans
		for i in range(0,i0.size):
			t0 = tsec0[i0[i]]
			t1 = tsec0[i1[i]-1]
			dt = t1 < t0 
			if dt > MaxGap:
				bad = np.where((tsec >= t0) & (tsec <= t1))[0]
				data[f][bad] = np.nan

	#calculate field magnitude
	data.Bm = np.sqrt(data.Bx**2 + data.By**2 + data.Bz**2)
	
	return data


def _filterData(data):
	
	cfg = profile.get()
	res = cfg.get('res',1.0)
	fmin = cfg['highPassFilter']
	fmax = cfg['lowPassFilter']
	

	#time axis
	utc0 = data.utc[0]
	tsec = (data.utc - utc0)*3600.0
	


	#filter the data
	if not fmax is None:
		print('Low-pass filtering B field')
		Filter = fmax
		data.Bx = ws.Filter.Filter(data.Bx,res,low=1/Filter)
		data.By = ws.Filter.Filter(data.By,res,low=1/Filter)
		data.Bz = ws.Filter.Filter(data.Bz,res,low=1/Filter)

		
	if not fmin is None:
		print('Removing Long-Period Background Stuff')
		lpBx = ws.Filter.Filter(data.Bx,res,low=1/fmin,KeepDC=False)
		lpBy = ws.Filter.Filter(data.By,res,low=1/fmin,KeepDC=False)
		lpBz = ws.Filter.Filter(data.Bz,res,low=1/fmin,KeepDC=False)
		data.Bx -= lpBx
		data.By -= lpBy
		data.Bz -= lpBz


	return data
	
	
def processMagData(data,Date,MaxGap=300.0):
	'''
	Take the magnetometer data and do some preprocessing (e.g. filtering,
	resampling)
	
	'''


	#fill gaps
	data = _fillDataGaps(data,MaxGap)
	
	#resample
	data = _resampleData(data,Date,MaxGap)
	
	#filter
	data = _filterData(data)
	
	return data
