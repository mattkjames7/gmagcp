import PyFileIO as pf
from .. import profile
import os

def readCrossPhase(date,estn,pstn):

    cfg = profile.get()
    pairPath = cfg['specPath'] + '/{:s}-{:s}'.format(estn,pstn)
    fname = pairPath + '/{:08d}.bin'.format(date)

    if os.path.isfile(fname):
        return pf.LoadObject(fname)
    else:
        return None