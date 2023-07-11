import numpy as np
import os
import PyFileIO as pf
import DateTimeTools as TT
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable



from ..Tools.FigText import FigText
import groundmag as gm
from scipy.interpolate import interp1d

from ..Tools.CheckPath import CheckPath
from .GetMagData import GetMagData
from .ProcessMagData import ProcessMagData
from .GetSpectrogram import GetSpectrogram
from .. import Globals
import wavespec as ws
from ..PlumeFP.PlotEqMagFP import PlotEqMagPairFP
from ..PlumeFP.TestEqFP import TestEqFP

from ..Tools.PlotCPEigenfreqs import PlotCPEigenfreqs
from ..Tools.kmeans import kmeans
from ..Tools.aggclust import aggclust

def Within(x,x0,x1):
	
	return ((x <= x0) & (x >= x1)) | ((x <= x1) & (x >= x0))



class CPCls(object):
	def __init__(self,estn,pstn,Date):
		self.estn = estn
		self.pstn = pstn
		self.date = Date
		self.fpath = Globals.DataPath + 'CP/Spec/{:s}-{:s}/'.format(estn.upper(),pstn.upper())	
		CheckPath(self.fpath)
		self.fname = self.fpath + '{:08d}.bin'.format(self.date)		
		
		if os.path.isfile(self.fname):
			self.__dict__ = pf.LoadObject(self.fname)
		else:
			try:
				# get the data for both stations
				self._GetData()
				
				self._FFT()
				
				self._CrossPhases()
				
				self._GetPos()
							
				pf.SaveObject(self.__dict__,self.fname)
				print('Saved: '+self.fname)
			except:
				print('Something went wrong')
				self.fail = True
				return None

	def _GetPos(self):
		
		self.px,self.py,self.pz = gm.Trace.MagPairTracePos(self.estn,self.pstn,self.Date,self.ut)
		st0 = gm.GetStationInfo(self.estn,self.date)
		st1 = gm.GetStationInfo(self.pstn,self.date)
		
		self.mlat = 0.5*(st0.mlat + st1.mlat)[0]
		self.mlon = 0.5*(st0.mlon + st1.mlon)[0]
		self.glat = 0.5*(st0.glat + st1.glat)[0]
		self.glon = 0.5*(st0.glon + st1.glon)[0]
			


	def _GetData(self):
		
		#get both sets of processed data
			
			
		print('Reading {:s} Data'.format(self.estn))
		edata0 = GetMagData(self.estn,self.date,3600)
		print('Processing {:s} Data'.format(self.estn))
		self.edata = ProcessMagData(edata0,self.date)
		print('Reading {:s} Data'.format(self.pstn))
		pdata0 = GetMagData(self.pstn,self.date,3600)
		print('Processing {:s} Data'.format(self.pstn))
		self.pdata = ProcessMagData(pdata0,self.date)


		
	def _FFT(self):


		espec = GetSpectrogram(self.edata,self.date)
		self.eFFT = espec['xFFT']
		pspec = GetSpectrogram(self.pdata,self.date)
		self.pFFT = pspec['xFFT']
			
			
		self.utc = espec['utc']	
		self.Date,self.ut = TT.ContUTtoDate(self.utc)
		self.utcax = espec['utcax']	
		self.freq = espec['freq']	
		self.freqax = espec['freqax']	
		self.Tspec = espec['tsec']

	
	def _CrossPhases(self):

		N0 = 3600.0/1.0
		cp = ws.DetectWaves.CPWavesFFTSpec(self.Tspec,self.freq,self.eFFT,self.pFFT,N0)
		self.cp = cp

	
						
	def Plot(self,Param='Cpha_smooth',ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],zlog=False,scale=None,
				cmap='Reds_r',zlabel='',nox=False,noy=False,ShowPP=True,ShowColorbar=True,PP='image',MaxDT=2.0):
		
		
		#get ut range
		uset = np.where((self.ut >= ut[0]) & (self.ut <= ut[1]))[0]
		t0 = uset[0]
		t1 = uset[-1] + 1
		utc = self.utcax[t0:t1+1]
				
		#and frequency range
		if flim is None:
			f0 = 0
			f1 = self.freq.size
			flim = np.array([self.freqax[0],self.freqax[-1]])*1000.0
		else:
			usef = np.where((self.freq*1000.0 >= flim[0]) & (self.freq*1000.0 <= flim[1]))[0]
			f0 = usef[0]
			f1 = usef[-1] + 1
		freq = self.freqax[f0:f1+1]*1000.0
		
		spec = self.cp[Param]
		spec = spec[t0:t1,f0:f1]	
		
		
		ax = self._Plot(utc,freq,spec,fig=fig,maps=maps,zlog=zlog,
				scale=scale,zlabel=zlabel,cmap=cmap,ShowColorbar=ShowColorbar)
		
		
		utclim = TT.ContUT(np.array([self.date,self.date]),np.array(ut))
		ax.set_xlim(utclim)
		ax.set_ylim(flim)
		
		if nox:
			ax.set_xlabel('')
			n = len(ax.get_xticks())
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('UT')
			TT.DTPlotLabel(ax)
		
		if noy:
			ax.set_ylabel('')
			n = len(ax.get_yticks())
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('$f$ (mHz)')
			
			
		if ShowPP:
			nc,_,_,_,cutc = TestEqFP(self.estn,self.pstn,self.date,ut=ut,PP=PP,MaxDT=MaxDT)
			if nc > 0:
				ax.vlines(cutc,flim[0],flim[1],color='black',linewidth=2)
				ax.vlines(cutc,flim[0],flim[1],color='lime',linestyle='--')
				
		
		title = '{:s}-{:s} mlat={:5.2f}, mlon={:5.2f}'.format(self.estn.upper(),self.pstn.upper(),self.mlat,self.mlon)
		FigText(ax,0.01,0.99,title,color='black',transform=ax.transAxes,ha='left',va='top')
		return ax


	def _Plot(self,xg,yg,grid,fig=None,maps=[1,1,0,0],zlog=False,
					scale=None,zlabel='',cmap=None,ShowColorbar=True,**kwargs):
		

		#get the scale
		if scale is None:
			if zlog:
				scale = [np.nanmin(grid[grid > 0]),np.nanmax(grid)]
			else:
				scale = [np.nanmin(grid),np.nanmax(grid)]
		
		#set norm
		if zlog:
			norm = colors.LogNorm(vmin=scale[0],vmax=scale[1])
		else:
			norm = colors.Normalize(vmin=scale[0],vmax=scale[1])	
		
		
		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
		else:
			ax = fig
			
		sm = ax.pcolormesh(xg,yg,grid.T,cmap=cmap,norm=norm)

		
		fig.subplots_adjust(right=0.9)
		
		if ShowColorbar:
			box = ax.get_position()
			cax = plt.axes([0.01*box.width + box.x1,box.y0+0.1*box.height,box.width*0.0125,box.height*0.8])
			cbar = fig.colorbar(sm,cax=cax)
			cbar.set_label(zlabel)
		
		return ax
		
		
	def GetSpectrum(self,ut,Param):
		
		
		utc = TT.ContUT(self.Date,ut)[0]
		dt = np.abs(utc - self.utc)
		I = np.argmin(dt)		
		
		spec = getattr(self,Param)
		
		return self.utc[I],self.freq,spec[I]
	
	
	def PlotSpectrum(self,ut,Param,flim=None,fig=None,maps=[1,1,0,0],
				nox=False,noy=False,ylog=False,label=None,dy=0.0):
		
		utc,freq,spec = self.GetSpectrum(ut,Param)
		
		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
		else:
			ax = fig
		
		ax.plot(freq*1000,spec+dy,label=label)
		
		if nox:
			ax.set_xlabel('')
			n = len(ax.get_xticks())
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('$f$ (mHz)')
		
		if noy:
			ax.set_ylabel('')
			n = len(ax.get_yticks())
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('')
		
		if ylog:
			ax.set_yscale('log')
		
		if flim:
			ax.set_xlim(flim)
		else:
			ax.set_xlim(self.freq[0]*1000.0,self.freq[-1]*1000.0)
			
		return ax		


	def PlotEigenfrequencies(self,ut=[0.0,24.0],**kwargs):
		'''
		Plot average eigenfrequencies using a subset of 
		crossphase spectra.
	
		'''
		
		if np.size(ut[0]) == 1:
			use = np.where((self.ut >= ut[0]) & (self.ut <= ut[1]))[0]
		else:
			keep = np.zeros(self.ut.size,dtype='bool')
			for u in ut:
				k = np.where((self.ut >= u[0]) & (self.ut <= u[1]))[0]
				keep[k] = True
			use = np.where(keep)[0]
		t = self.ut[use]
		f = self.freq
		P = self.cp['Cpha_smooth'][use]
		return PlotCPEigenfreqs(f,P,**kwargs)
