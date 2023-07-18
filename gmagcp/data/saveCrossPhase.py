import numpy as np
from .getMagData import getMagData
from .processMagData import processMagData
from .getSpectrogram import getSpectrogram
from .. import profile
from ..tools.checkPath import checkPath
import PyFileIO as pf


def _processData(date,estn,pstn):

    cfg = profile.get()
    window = cfg['window']

    print('Reading {:s} Data'.format(estn))
    edata0 = getMagData(estn,date,window)
    print('Processing {:s} Data'.format(estn))
    edata = processMagData(edata0,date,Window=cfg['window'],Fmin=cfg['highPassFilter'],Fmax=cfg['lowPassFilter'])
    print('Reading {:s} Data'.format(pstn))
    pdata0 = getMagData(pstn,date,window)
    print('Processing {:s} Data'.format(pstn))
    pdata = processMagData(pdata0,window,Window=cfg['window'],Fmin=cfg['highPassFilter'],Fmax=cfg['lowPassFilter'])


    return edata,pdata

def _fftData(date,edata,pdata):

    cfg = profile.get()

    print('Equatorward Spectrogram')
    espec = getSpectrogram(edata,date,window=cfg['window'],slip=cfg['slip'],
                           Freq0=cfg['freq0'],Freq1=cfg['freq1'],detrend=cfg['detrend'])
    eFFT = espec['xFFT']

    print('Poleward Spectrogram')
    pspec = getSpectrogram(pdata,date,window=cfg['window'],slip=cfg['slip'],
                           Freq0=cfg['freq0'],Freq1=cfg['freq1'],detrend=cfg['detrend'])
    pFFT = pspec['xFFT']

    return eFFT,pFFT


def _cpData(efft,pfft):

    cfg = profile.get()
    print('Cross Phase Spectra')
    N0 = cfg['window']/1.0
    cp = ws.DetectWaves.CPWavesFFTSpec(efft['Tspec'],efft['freq'],efft,pfft,N0)
    return cp


def _magPos(date,estn,pstn):

    print('Mag Pos')
    px,py,pz = gm.Trace.MagPairTracePos(estn,pstn,estn['Date'],estn['ut'])
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
    efft,pfft = _fftData(date,edata,pdata)

    #cross phase
    cp = _cpData(efft,pfft)

    #get the magnetometer position for tracing
    pos = _magPos(date,estn,pstn)

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

    pairPath = cfg['path'] + '/{:s}-{:s}'.format(estn,pstn)
    if not os.path.isdir(pairPath):
        os.makedirs(pairPath)

    fname = pairPath + '/{:08d}.bin'.format(date)
    pf.SaveObject(out,fname)
    print('Saved: ',fname)