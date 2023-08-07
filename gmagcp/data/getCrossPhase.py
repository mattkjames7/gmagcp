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
            n += d['efft'].shape[0]

    out['efft'] = np.zeros((n,d['efft'].shape[1]),dtype='complex128')
    out['pfft'] = np.zeros((n,d['efft'].shape[1]),dtype='complex128')

    p = 0
    for d in dataList:
        if d is not None:
            l = d['efft'].shape[0]
            out['efft'][p:p+l] = d['efft']
            out['pfft'][p:p+l] = d['pfft']
            p += l

    
    for d in dataList:
        if d is not None:
            cpkeys = list(d['cp'].keys())
            break
    
    cp = {}
    for key in cpkeys:
        tmp = [d['cp'][key] for d in dataList if d is not None]
        if len(tmp) > 0:
            sh = tmp[0].shape
            if key == 'F':
                cp[key] = tmp[0]
            elif key == 'Tax':
                cp[key] = np.concatenate([t[:-1] for t in tmp]+[[tmp[-1][-1]]])
            elif len(sh) == 1:
                cp[key] = np.concatenate(tmp)
            else:
                cp[key] = np.concatenate(tmp,axis=0)
    out['cp'] = cp

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

    data = [readCrossPhase(d,estn,pstn) for d in dates if d is not None]
    if len(data) == 0:
        return None

    return _combineData(data)