import numpy as np
from .getMagData import getMagData
from .processMagData import processMagData
from .getSpectrogram import getSpectrogram
from .. import profile
from ..tools.checkPath import checkPath
import PyFileIO as pf
import wavespec as ws
import groundmag as gm
import DateTimeTools as TT
import os

def _processData(date,estn,pstn):

    cfg = profile.get()
    window = cfg['window']

    print('Reading {:s} Data'.format(estn))
    edata0 = getMagData(estn,date)
    print('Processing {:s} Data'.format(estn))
    edata = processMagData(edata0,date)
    print('Reading {:s} Data'.format(pstn))
    pdata0 = getMagData(pstn,date)
    print('Processing {:s} Data'.format(pstn))
    pdata = processMagData(pdata0,date)

    return edata,pdata

def _fftData(date,edata,pdata):

    cfg = profile.get()

    print('Equatorward Spectrogram')
    espec = getSpectrogram(edata,date)
    eFFT = espec['xFFT']

    print('Poleward Spectrogram')
    pspec = getSpectrogram(pdata,date)
    pFFT = pspec['xFFT']

    tSpec = espec['utc']
    freq = espec['freq']

    return tSpec,freq,eFFT,pFFT


def _cpData(tSpec,freq,efft,pfft):

    cfg = profile.get()
    print('Cross Phase Spectra')
    N0 = cfg['window']/1.0
    cp = ws.DetectWaves.CPWavesFFTSpec(tSpec,freq,efft,pfft,N0)
    return cp


def _magPos(date,estn,pstn,tspec):

    Date,ut = TT.ContUTtoDate(tspec)
    print('Mag Pos')
    px,py,pz = gm.Trace.MagPairTracePos(estn,pstn,Date,ut)
    st0 = gm.GetStationInfo(estn,date)
    st1 = gm.GetStationInfo(pstn,date)

    mlat = 0.5*(st0.mlat + st1.mlat)[0]
    mlon = 0.5*(st0.mlon + st1.mlon)[0]
    glat = 0.5*(st0.glat + st1.glat)[0]
    glon = 0.5*(st0.glon + st1.glon)[0]

    out = {
        'px' : px,
        'py' : py,
        'pz' : pz,
        'mlat' : mlat,
        'mlon' : mlon,
        'glat' : glat,
        'glon' : glon,
    }
    return out


def saveCrossPhase(date,estn,pstn):

    #process the data
    edata,pdata = _processData(date,estn,pstn)

    #fft the data
    tSpec,freq,efft,pfft = _fftData(date,edata,pdata)

    #cross phase
    cp = _cpData(tSpec,freq,efft,pfft)

    #get the magnetometer position for tracing
    pos = _magPos(date,estn,pstn,tSpec)

    out = {
        'edata' : edata,
        'pdata' : pdata,
        'efft' : efft,
        'pfft' : pfft,
        'cp' : cp,
        'pos' : pos
    }

    #save
    cfg = profile.get()
    checkPath(cfg['specPath'])

    pairPath = cfg['specPath'] + '/{:s}-{:s}'.format(estn,pstn)
    if not os.path.isdir(pairPath):
        os.makedirs(pairPath)

    fname = pairPath + '/{:08d}.bin'.format(date)
    pf.SaveObject(out,fname)
    print('Saved: ',fname)