import numpy as np
import json
from . import globs
import os

default = {
    'name' : 'default',
    'method' : 'fft',
    'window' : 1800.0,
    'slip' : 600.0,
    'freq0' : 0.0,
    'freq1' : 0.1,
    'detrend' : 2,
    'lowPassFilter' : 0.0,
    'highPassFilter' : 0.0,
    'windowFunc' : 'none',
}

def create(name='default',**kwargs):

    out = {}
    for k in default:
        out[k] = default[k]
    for k in kwargs:
        out[k] = kwargs[k]


    profilePath = globs.dataPath + '/profiles/{:s}'.format(name)

    if os.path.isdir(profilePath):
        print('Profile appears to exist, returning...')
        return
    else:
        os.makedirs(profilePath)

    out['path'] = profilePath
    out['specPath'] = profilePath + '/spec'
    

    fname = profilePath + '/config.cfg'
    with open(fname,'w') as f:
        json.dump(out,f,indent=2)
    print('Saved: {:s}'.format(fname))
    

def read(name='default'):

    profilePath = globs.dataPath + '/profiles/{:s}'.format(name)
    fname = profilePath + '/config.cfg'

    if os.path.isfile(fname):

        with open(fname,'r') as f:
            out = json.load(f)

        return out
    else:
        return None
    
def use(name='default'):

    cfg = read(name)
    if cfg == None:
        print('{:s} profile does not exist'.format(name))
        if name == 'default':
            create()
            cfg = read(name)
        else:
            print('Returning...')
            return

    print('Using profile: {:s}'.format(name))
    globs.profile = cfg

def get(name=None):

    if name == None:
        return globs.profile
    else:
        return read(name)