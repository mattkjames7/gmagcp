import numpy as np
import DateTimeTools as TT
import groundmag as gm
import wavespec as ws

def _getWindows(data,Date,window=3600,slip=300,Res=1.0):
	
	#get current time axis in seconds
	utc0 = TT.ContUT(Date,0.0)[0]
	tsec = (data.utc - utc0)*3600.0
	
	
	#get the number of windows using the imte range
	T0 = -window
	T1 = 86400 + window
	
	nw = np.int32(np.round((T1 - T0 - window)/slip)) + 1
	
	#get the indices of each start and end element of each window
	#(split this up before doing LS or Res != 1.0)
	t0 = np.arange(nw)*slip - window
	t1 = t0 + window
	tc = np.float64(t0) + window/2.0
	
	i0 = np.int32(np.round((t0 + window)/Res))
	i1 = np.int32(np.round((t1 + window)/Res))

	
	#now the groups of windows for coherence
	nc = np.int32(np.round(window//slip)) + 1
	cnw = nw - nc + 1
	ci0 = np.arange(cnw)
	ci1 = ci0 + nc
	ctc = tc[nc//2:-(nc//2+1)]
	
	return nw,i0,i1,tc,cnw,ci0,ci1,ctc
	
def _getFrequencies(window=3600,Res=1.0):
	
	freq = np.arange(window+1,dtype='float64')/(np.float32(window*Res))
	nf = np.size(freq) - 1
	nf = nf//2
	freq = freq[:nf]		

	return nf,freq
	
def _magPos(stn,t_utc):
	
	stn = gm.GetStationInfo(stn)
	glat = stn.glat[0]*np.pi/180.0
	glon = stn.glon[0]*np.pi/180.0
	t_date,t_ut = TT.ContUTtoDate(t_utc)
	g_z = np.zeros(t_utc.size) + 1.02*np.sin(glat)
	g_x = np.zeros(t_utc.size) + 1.02*np.cos(glon)*np.cos(glat)
	g_y = np.zeros(t_utc.size) + 1.02*np.sin(glon)*np.cos(glat)

	return g_x,g_y,g_z
	
	
def _getKVectors(xfft,yfft,zfft):


	Jxy = xfft.imag*yfft.real - yfft.imag*xfft.real
	Jxz = xfft.imag*zfft.real - zfft.imag*xfft.real
	Jyz = yfft.imag*zfft.real - zfft.imag*yfft.real
	A = np.sqrt(Jxy**2 + Jxz**2 + Jyz**2)	
	kx = Jyz/A
	ky =-Jxz/A
	kz = Jxy/A		
	
	return kx,ky,kz	

def getSpectrogram(data,Date,window=3600,slip=300,Res=1.0,Freq0=0.0,Freq1=0.05):


	tsec = (data.utc - TT.ContUT(Date,0.0)[0])*3600.0
	
	#calculate windows
	nw,i0,i1,tc,cnw,ci0,ci1,ctc = _getWindows(data,Date,window,slip,Res)
	
	#get the frequencies
	nf,freq = _getFrequencies()

	#get the Fourier spectra
	fft = {}
	Bfield = {}
	fields = ['Bx']
	comps = ['x']

	for f,c in zip(fields,comps):

		fft[c+'FFT'] = np.zeros((nw,nf),dtype='complex128')
		Bfield[f] = np.zeros((nw,),dtype='float64')
		for i in range(0,nw):
			ind = np.arange(i0[i],i1[i])
			t = tsec[ind]
			b = ws.Tools.PolyDetrend(t,data[f][ind],2)

			Bfield[f][i] = np.nanmean(data[f][ind])
			
			pw,am,ph,fr,fi,_ = ws.Fourier.FFT(t,b,OneSided=True)

			fft[c+'FFT'][i] = fr + 1.0j*fi

	#limit frequency range
	usef = np.where((freq >= Freq0) & (freq <= Freq1))[0]
	keys = list(fft.keys())
	for k in keys:
		fft[k] = fft[k][:,usef]
	freq = freq[usef]
	nf = freq.size

	#limit to stuff from this date
	usefft = np.where((tc >= 0.0) & (tc <86400))[0]
	
	#output array
	spec = {}
	spec['nw'] = usefft.size
	spec['i0'] = i0
	spec['i1'] = i1
	spec['tsec'] = tc[usefft]
	spec['utc'] = TT.ContUT(Date,0.0)[0] + tc[usefft]/3600.0
	spec['utcax'] = np.append(spec['utc']-0.5*slip/3600.0,spec['utc'][-1] + 0.5*slip/3600.0)
	spec['nf'] = nf
	spec['freq'] = freq
	df = freq[1]
	spec['freqax'] = np.append(freq,freq[-1]+df)
	
	keys = list(fft.keys())
	for k in keys:
		spec[k] = fft[k][usefft]

		
	return spec
