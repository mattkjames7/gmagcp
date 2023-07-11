import numpy as np
import DateTimeTools as TT
import groundmag as gm

def getMagData(stn,Date,Window=0.0):
	'''
	Get the magnetometer data from a specific date.
	
	'''
	

	#find the window length
	hw = Window/3600
	
	#get the time limits	
	Date0 = TT.MinusDay(Date)
	Date1 = TT.PlusDay(Date)
	ut0 = 24.0 - hw
	ut1 = hw
	
	#now the data
	data = gm.GetData(stn,[Date0,Date1],ut=[ut0,ut1])
	
	
	return data
