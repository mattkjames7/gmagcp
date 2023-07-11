import numpy as np
from .readCrossPhase import readCrossPhase
import DateTimeTools as TT


def _combineData(dataList):

    out = {}

    n = 0
    for d in dataList:
        if d is not None:
            dtype = d['edata'].dtype
            n += d['edata'].size
    
    if n == 0:
        return None

    out['edata'] = np.recarray(n,dtype=dtype)
    out['pdata'] = np.recarray(n,dtype=dtype)

    p = 0
    for d in dataList:
        if d is not None:
            l = d['edata'].size
            out['edata'][p:p+l] = d['edata']
            out['pdata'][p:p+l] = d['pdata']
            p += l

    n = 0
    for d in dataList:
        if d is not None:
            dtype = d['efft']
            n += d['efft'].size

    out['efft'] = np.recarray(n,dtype=dtype)
    out['pfft'] = np.recarray(n,dtype=dtype)

    p = 0
    for d in dataList:
        if d is not None:
            l = d['efft'].size
            out['efft'][p:p+l] = d['efft']
            out['pfft'][p:p+l] = d['pfft']
            p += l

    out['cp'] = {}
    for d in dataList:
        if d is not None:
            for k in d:
                if k not in out['cp']:
                    out['cp'][k] = []
                out['cp'][k].append(d[k])

    for k in out['cp']:
        out['cp'] = np.concatenate(out['cp'][k])

    for d in dataList:
        if d is not None:
            out['pos'] = {
                'mlon' : d['pos']['mlon'],
                'mlat' : d['pos']['mlat'],
                'glon' : d['pos']['glon'],
                'glat' : d['pos']['glat'],
            }
            break

    pos = ['px','py','pz']
    for p in pos:
        out['pos'][p] = np.concatenate([d['pos'][p] for d in dataList])

    return out

def getCrossPhase(date,estn,pstn):

    ndate = np.size(date)
    if np.size(date) != 2:
        dates = np.zeros(ndate,dtype='int32') + date
    else:
        dates = TT.ListDates(date[0],date[1])

    data = [readCrossPhase(d,estn,pstn) for d in dates]

    return _combineData(data)