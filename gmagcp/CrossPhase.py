import numpy as np
import os
import PyFileIO as pf
import DateTimeTools as TT
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..tools.figText import figText
import groundmag as gm
from .data.getMagData import getMagData
from .data.processMagData import processMagData
from .data.getSpectrogram import getSpectrogram
from .data.readCrossPhase import readCrossPhase


class CrossPhase(object):
	def __init__(self,estn,pstn,Date,ut=[0.0,24.0]):
		self.estn = estn
		self.pstn = pstn
		self.date = Date
		self.ut = ut
			
		

		try:
			self.data = readCrossPhase()
			for k in self.data:
				setattr(self,k,self.data[k])
		except:
			print('Something went wrong')
			self.fail = True
			return None

						
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
		figText(ax,0.01,0.99,title,color='black',transform=ax.transAxes,ha='left',va='top')
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
