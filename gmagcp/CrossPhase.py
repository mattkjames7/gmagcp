import numpy as np
import os
import PyFileIO as pf
import DateTimeTools as TT
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .tools.figText import figText
import groundmag as gm
from .data.getCrossPhase import getCrossPhase
from . import profile
from scipy.signal import savgol_filter
from .data.saveCrossPhase import saveCrossPhase


class CrossPhase(object):
	def __init__(self,Date,estn,pstn):
		self.estn = estn
		self.pstn = pstn
		self.date = Date
			
		

		try:
			self.data = getCrossPhase(Date,estn,pstn)
			for k in self.data:
				setattr(self,k,self.data[k])
			self.freq = self.cp['F']
			df = self.freq[1] - self.freq[0]
			self.freqax = np.append(self.freq,self.freq[-1]+df)
			self.tspec = self.cp['Tspec']/3600.0
			windowh = profile.get()['window']/7200.0
			self.tax = np.append(self.tspec-windowh,self.tspec[-1]+windowh)
		except Exception as e:
			print('Something went wrong')
			print(e)
			self.fail = True
			return None

						
	def plot(self,Param='Cpha_smooth',date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],zlog=False,scale=None,
				cmap='seismic_r',zlabel='',nox=False,noy=False,showColorbar=True,showEigenFreqs=True,colspan=1,rowspan=1):
		
		
		if date is None:
			date = [np.min(self.date),np.max(self.date)]
		

		#get ut range
		ut0,ut1 = TT.ContUT(date,ut)

		uset = np.where((self.tspec >= ut0) & (self.tspec <= ut1))[0]
		t0 = uset[0]
		t1 = uset[-1] + 1
		utc = self.tax[t0:t1+1]


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
		
		
		ax = self._plot(utc,freq,spec,fig=fig,maps=maps,zlog=zlog,
				scale=scale,zlabel=zlabel,cmap=cmap,showColorbar=showColorbar,colspan=colspan,rowspan=rowspan)
		
		
		utclim = TT.ContUT(np.array([self.date,self.date]),np.array(ut))
		ax.set_xlim(utclim)
		ax.set_ylim(flim)
		
		if nox:
			ax.set_xlabel('')
			xt = ax.get_xticks()
			ax.set_xticks(xt)
			n = len(xt)
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('UT')
			TT.DTPlotLabel(ax)
		
		if noy:
			ax.set_ylabel('')
			yt = ax.get_yticks()
			ax.set_yticks(yt)
			n = len(yt)
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('$f$ (mHz)')
		ax.set_xlim(utclim)
			
		if showEigenFreqs:
			ax.scatter(self.cp['t']/3600.0,self.cp['f']*1000,color='black',zorder=2,marker='+')
				
		
		title = '{:s}-{:s} mlat={:5.2f}, mlon={:5.2f}'.format(self.estn.upper(),self.pstn.upper(),self.pos['mlat'],self.pos['mlon'])
		figText(ax,0.01,0.99,title,color='black',transform=ax.transAxes,ha='left',va='top')
		return ax

	def plotFFTe(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],zlog=True,scale=None,
				cmap='gnuplot',nox=False,noy=False,showColorbar=True,showEigenFreqs=True,colspan=1,rowspan=1):

		if date is None:
			date = [np.min(self.date),np.max(self.date)]
		

		#get ut range
		ut0,ut1 = TT.ContUT(date,ut)

		uset = np.where((self.tspec >= ut0) & (self.tspec <= ut1))[0]
		t0 = uset[0]
		t1 = uset[-1] + 1
		utc = self.tax[t0:t1+1]


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
		
		spec = np.abs(self.efft)**2
		spec = spec[t0:t1,f0:f1]	
		if scale is None:
			if zlog:
				scale = [np.nanmin(spec[spec > 0]),np.nanmax(spec)]
			else:
				scale = [np.nanmin(spec),np.nanmax(spec)]

		ax = self._plot(utc,freq,spec,fig=fig,maps=maps,zlog=zlog,
				scale=scale,zlabel='Power\n{:s}'.format(self.estn),cmap=cmap,showColorbar=showColorbar,colspan=colspan,rowspan=rowspan)
		
		
		utclim = TT.ContUT(np.array([self.date,self.date]),np.array(ut))
		ax.set_xlim(utclim)
		ax.set_ylim(flim)
		
		if nox:
			ax.set_xlabel('')
			xt = ax.get_xticks()
			ax.set_xticks(xt)
			n = len(xt)
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('UT')
			TT.DTPlotLabel(ax)
		
		if noy:
			ax.set_ylabel('')
			yt = ax.get_yticks()
			ax.set_yticks(yt)
			n = len(yt)
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('$f$ (mHz)')
		ax.set_xlim(utclim)
			
		if showEigenFreqs:
			ax.scatter(self.cp['t']/3600.0,self.cp['f']*1000,color='black',zorder=2,marker='+')
				
		
		title = '{:s}-{:s} mlat={:5.2f}, mlon={:5.2f}'.format(self.estn.upper(),self.pstn.upper(),self.pos['mlat'],self.pos['mlon'])
		figText(ax,0.01,0.99,title,color='black',transform=ax.transAxes,ha='left',va='top')
		return ax


	def plotFFTp(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],zlog=True,scale=None,
				cmap='gnuplot',nox=False,noy=False,showColorbar=True,showEigenFreqs=True,colspan=1,rowspan=1):

		if date is None:
			date = [np.min(self.date),np.max(self.date)]
		

		#get ut range
		ut0,ut1 = TT.ContUT(date,ut)

		uset = np.where((self.tspec >= ut0) & (self.tspec <= ut1))[0]
		t0 = uset[0]
		t1 = uset[-1] + 1
		utc = self.tax[t0:t1+1]


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
		
		spec = np.abs(self.pfft)**2
		spec = spec[t0:t1,f0:f1]	
		if scale is None:
			if zlog:
				scale = [np.nanmin(spec[spec > 0]),np.nanmax(spec)]
			else:
				scale = [np.nanmin(spec),np.nanmax(spec)]

		ax = self._plot(utc,freq,spec,fig=fig,maps=maps,zlog=zlog,
				scale=scale,zlabel='Power\n{:s}'.format(self.pstn),cmap=cmap,showColorbar=showColorbar,colspan=colspan,rowspan=rowspan)
		
		
		utclim = TT.ContUT(np.array([self.date,self.date]),np.array(ut))
		ax.set_xlim(utclim)
		ax.set_ylim(flim)
		
		if nox:
			ax.set_xlabel('')
			xt = ax.get_xticks()
			ax.set_xticks(xt)
			n = len(xt)
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('UT')
			TT.DTPlotLabel(ax)
		
		if noy:
			ax.set_ylabel('')
			yt = ax.get_yticks()
			ax.set_yticks(yt)
			n = len(yt)
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('$f$ (mHz)')
		ax.set_xlim(utclim)
			
		if showEigenFreqs:
			ax.scatter(self.cp['t']/3600.0,self.cp['f']*1000,color='black',zorder=2,marker='+')
				
		
		title = '{:s}-{:s} mlat={:5.2f}, mlon={:5.2f}'.format(self.estn.upper(),self.pstn.upper(),self.pos['mlat'],self.pos['mlon'])
		figText(ax,0.01,0.99,title,color='black',transform=ax.transAxes,ha='left',va='top')
		return ax


	def _plot(self,xg,yg,grid,fig=None,maps=[1,1,0,0],zlog=False,
					scale=None,zlabel='',cmap=None,showColorbar=True,colspan=1,rowspan=1,**kwargs):
		

		#get the scale
		if scale is None:
			if zlog:
				scale = [np.nanmin(grid[grid > 0]),np.nanmax(grid)]
			else:
				scale = [np.nanmin(grid),np.nanmax(grid)]
				smx = np.nanmax(np.abs(scale))
				scale = [-smx,smx]
		
		#set norm
		if zlog:
			norm = colors.LogNorm(vmin=scale[0],vmax=scale[1])
		else:
			norm = colors.Normalize(vmin=scale[0],vmax=scale[1])	
		
		
		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]),colspan=colspan,rowspan=rowspan)
		else:
			ax = fig
			
		sm = ax.pcolormesh(xg,yg,grid.T,cmap=cmap,norm=norm)

		
		fig.subplots_adjust(right=0.85)
	
		if showColorbar:
			box = ax.get_position()
			cax = plt.axes([0.01*box.width + box.x1,box.y0+0.1*box.height,box.width*0.0125,box.height*0.8])
			cbar = fig.colorbar(sm,cax=cax)
			cbar.set_label(zlabel)
		
		return ax
		
		
	def getSpectrum(self,date,ut,Param):
		
		
		utc = TT.ContUT(date,ut)[0]
		dt = np.abs(utc - self.tspec)
		I = np.argmin(dt)		
		
		spec = self.cp.get(Param,self.cp['Cpha_smooth'])
		
		return self.tspec[I],self.freq,spec[I]
	
	
	def plotSpectrum(self,date,ut,Param,flim=None,fig=None,maps=[1,1,0,0],
				nox=False,noy=False,ylog=False,label=None,dy=0.0):
		
		utc,freq,spec = self.getSpectrum(date,ut,Param)
		
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


	def plotEigenfrequencies(self,date=None,ut=[0.0,24.0],**kwargs):
		'''
		Plot average eigenfrequencies using a subset of 
		crossphase spectra.
	
		'''
		
		if date is None:
			t0 = self.tspec[0]
			t1 = self.tspec[-1]
		else:
			t0,t1 = TT.ContUT([np.min(date),np.max(date)],ut)
			

		use = np.where((self.tspec >= t0) & (self.tspec <= t1))[0]

		t = self.tspec[use]
		f = self.freq
		P = self.cp['Cpha_smooth'][use]
		return self._plotCPEigenfreqs(f,P,**kwargs)



	def _plotCPEigenfreqs(self,f,P,fig=None,maps=[1,1,0,0],nox=False,
						legend=True,ShowPeaks=True,ShowTroughs=True,Class=None):
		'''
		f : float
			Frequency array, Hz
		P : float
			CrossPhase (nt,nf) nt: number of time elements, nf: number of frequency elements.

		'''
		
		
		
		#calculate mean and stdev
		mu = np.nanmean(P,axis=0)
		sd = np.nanstd(P,axis=0)

		#calculate frequency array
		F = 1000.0*f
		
		
		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
		else:
			ax = fig

		ax.fill(np.append(F,F[::-1]),np.append(mu + sd,(mu - sd)[::-1]),color='grey',label=r'$\mu \pm \sigma$')
		ax.plot(F,mu,color='red',label=r'$\mu$')
		ax.plot([F[0],F[-1]],[0.0,0.0],color='black',linestyle='--')
		
		
		mu = savgol_filter(mu,11,3)
		
		if ShowPeaks:
			
			#find peaks
			pk = np.where((mu[1:-1] > mu[:-2]) & (mu[1:-1] > mu[2:]) & (mu[1:-1] > 5.0))[0]
			if pk.size > 0:
				pk = pk + 1
				yl = ax.get_ylim()
				ym = np.mean(yl)
				ax.set_ylim(yl)
				fpk = F[pk]
				ax.vlines(fpk,yl[0],yl[1],color='black',zorder=2,linestyle=':')
				for f in fpk:
					ax.text(f,ym,'$f$={:4.1f}'.format(f),rotation=90.0,ha='left',va='center')
		if ShowTroughs:
			#find peaks
			pk = np.where((mu[1:-1] < mu[:-2]) & (mu[1:-1] < mu[2:]) & (mu[1:-1] < -5.0))[0]
			if pk.size > 0:
				pk = pk + 1
				yl = ax.get_ylim()
				ym = np.mean(yl)
				ax.set_ylim(yl)
				fpk = F[pk]
				ax.vlines(fpk,yl[0],yl[1],color='black',zorder=2,linestyle=':')
				for f in fpk:
					ax.text(f,ym,'$f$={:4.1f}'.format(f),rotation=90.0,ha='left',va='center')
		
		
		if legend:		
			ax.legend()
		
		if nox:
			ax.set_xticks(ax.get_xticks())
			ax.set_xticklabels(np.size(ax.get_xticks())*[''])
		else:
			ax.set_xlabel('Frequency (mHz)')
		ax.set_xlim(F[0],F[-1])
		ax.set_ylabel(r'$\times$Phase ($^\circ$)')
		if not Class is None:
			ax.text(0.01,0.99,'Class {:d}'.format(Class+1),ha='left',va='top',transform=ax.transAxes)

		return ax


	def plotCPlargest(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='Cpha_largest',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[-90,90],
			cmap='seismic_r',
			zlabel='Dominant\nx-phase',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	


	def plotCPsmooth(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='Cpha_smooth',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[-90,90],
			cmap='seismic_r',
			zlabel='Smoothed\nx-phase',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	

	def plotCP(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='Cpha',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[-90,90],
			cmap='seismic_r',
			zlabel='x-phase',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	
	
	def plotCPfinal(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=True):


		ax = self.plot(
			Param='Cpha_surv',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[-90,90],
			cmap='seismic_r',
			zlabel='Final\nx-phase',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	
	def plotPowerRatio(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='PR',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[0,4],
			cmap='Greens',
			zlabel='Power\nRatio',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	
	
	def plotTstat(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='ttest',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=False,
			scale=[0,2],
			cmap='Greys',
			zlabel='t-statisitic',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	
	def plotCrossPower(self,date=None,ut=[0.0,24.0],flim=None,fig=None,maps=[1,1,0,0],
			nox=False,noy=False,showColorBar=True,showEigenFreqs=False):


		ax = self.plot(
			Param='Cpow',
			date=date,
			ut=ut,
			flim=flim,
			fig=fig,
			maps=maps,
			zlog=True,
			scale=[1e-2,1e4],
			cmap='Blues',
			zlabel='x-power',
			nox=nox,
			noy=noy,
			showColorbar=showColorBar,
			showEigenFreqs=showEigenFreqs
		)
		return ax
	
	def plotBx(self,date=None,ut=[0.0,24.0],fig=None,maps=[1,1,0,0],nox=False,noy=False,dB=200.0):


		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
		else:
			ax = fig


		emu = np.nanmean(self.edata.Bx)
		pmu = np.nanmean(self.pdata.Bx)


		ax.plot(self.edata.utc,self.edata.Bx-emu,color='black')
		ax.plot(self.pdata.utc,self.pdata.Bx-pmu+dB,color='black')


		if date is None:
			date = [np.min(self.date),np.max(self.date)]

		utclim = TT.ContUT(np.array([self.date,self.date]),np.array(ut))
		ax.set_xlim(utclim)

		xt = 1.02*(utclim[1] - utclim[0]) + utclim[0]
		ax.text(xt,0.0,self.estn,color='black',clip_on=False)
		ax.text(xt,dB,self.pstn,color='black',clip_on=False)


		if nox:
			ax.set_xlabel('')
			xt = ax.get_xticks()
			ax.set_xticks(xt)
			n = len(xt)
			lab = ['']*n
			ax.set_xticklabels(lab)
		else:
			ax.set_xlabel('UT')
			TT.DTPlotLabel(ax)
		
		if noy:
			ax.set_ylabel('')
			yt = ax.get_yticks()
			ax.set_yticks(yt)
			n = len(yt)
			lab = ['']*n
			ax.set_yticklabels(lab)
		else:
			ax.set_ylabel('$B_h$ (nT)')
		ax.set_xlim(utclim)

		return ax
	

	def plotPage(self,date=None,ut=[0.0,24.0]):
		
		plt.figure(figsize=(8,11))
		fig = plt
		ax0 = self.plotBx(fig=fig,maps=[1,8,0,0],nox=True)
		ax1 = self.plotCrossPower(fig=fig,maps=[1,8,0,1],nox=True)
		ax2 = self.plotCP(fig=fig,maps=[1,8,0,2],nox=True)
		ax3 = self.plotPowerRatio(fig=fig,maps=[1,8,0,3],nox=True)
		ax4 = self.plotCPsmooth(fig=fig,maps=[1,8,0,4],nox=True)
		ax5 = self.plotCPlargest(fig=fig,maps=[1,8,0,5],nox=True)
		ax6 = self.plotTstat(fig=fig,maps=[1,8,0,6],nox=True)
		ax7 = self.plotCPfinal(fig=fig,maps=[1,8,0,7])

		if date is None:
			date = [np.min(self.date),np.max(self.date)]
		plt.subplots_adjust(top=0.93)
		h0,m0,_,_ = TT.DectoHHMM(ut[0])
		h1,m1,_,_ = TT.DectoHHMM(ut[1])
		plt.suptitle('{:s}-{:s}\n{:08d}T{:02d}:{:02d} - {:08d}T{:02d}:{:02d}'.format(self.estn,self.pstn,date[0],h0[0],m0[0],date[1],h1[0],m1[0]))
		