import PyFileIO as pf
from .. import profile
import os
from .saveCrossPhase import saveCrossPhase

def readCrossPhase(date,estn,pstn,autoSave=True):

    cfg = profile.get()
    pairPath = cfg['specPath'] + '/{:s}-{:s}'.format(estn,pstn)
    fname = pairPath + '/{:08d}.bin'.format(date)

    if os.path.isfile(fname):
        if os.stat(fname).st_size == 0:
            return None
        return pf.LoadObject(fname)
    else:
        print('Pair {:s}-{:s} on {:d} does not yet exist, attempting to recreate...'.format(estn,pstn,date))
        saveCrossPhase(date,estn,pstn)
        return readCrossPhase(date,estn,pstn)