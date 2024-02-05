# gmagcp
Ground magnetometer cross phase.

## Install 

This package should be built as a wheel or source package.

Firstly, clone and enter the cloned directory:

```bash
git clone git@github.com:mattkjames7/gmagcp.git
cd gmagcp
```

To build as a wheel:

```bash
python3 setup.py bdist_wheel
pip3 install dist/gmagcp-0.0.2-py3-none-any.whl
```

Or as source package:
```bash
python3 setup.py sdist
pip3 install dist/gmagcp-0.0.2.tar.gz
```

Set `$GMAGCP_PATH` to point to a directory which will contain cross-phase data
and profiles for configuring the package. This can be done in `~./bashrc`, e.g.:
```bash
export GMAGCP_PATH=/path/to/cp
```


## Usage

### Profiles

Profiles are used to define the cross phase parameters, the defaults are stored
in a `dict` below:
```python
parameters = {
    'name' : 'default',             #name of the profile
    'method' : 'fft',               # 'fft' or possibly 'ls'
    'window' : 2400.0,              # window size in seconds
    'slip' : 300.0,                 # how far the window moves in seconds
    'freq0' : 0.0,                  # lowest frequency to keep (Hz)
    'freq1' : 0.05,                 # highest frequency to keep (Hz)
    'detrend' : 2,                  # order of polynomial to detrend with
    'lowPassFilter' : None,         # low pass frequency cutoff
    'highPassFilter' : 0.000625,    # high pass frequency cutoff
    'windowFunc' : 'none',          # name of window function, e.g. "hann"
}
```

A new profile can be created using `gmagcp.profile.create()`, e.g.:
```python
# create a new profile based on default
gmagcp.profile.create("profile0",slip=400.0)

# or a completely new one
parameters = {
    'name' : 'profile1',           
    'method' : 'fft',             
    'window' : 3600.0,             
    'slip' : 1200.0,               
    'freq0' : 0.001,            
    'freq1' : 0.020,              
    'detrend' : 1,                
    'lowPassFilter' : 0.02,       
    'highPassFilter' : 0.001,   
    'windowFunc' : 'hann',     
gmagcp.profile.create(**parameters)
```

Obtain the parameters of an existing one by name:
```python
params = gmagcp.read.read("profile0")
```

Set the current profile to use:
```python
gmagcp.profiles.use("profile1")
```

Retrieve the profile currently in use:
```python
parameters = gmagcp.profiles.get()
```


### Saving and reading cross-phases

Saving and reading cross phases uses the currently loaded profile. Both methods
take the following arguments:
 - `Date` : date in the format `yyyymmdd`
 - `estn` : equatorward station code
 - `pstn` : poleward station code

For saving the cross-phase:
```python
gmagcp.data.saveCrossPhase(20200101,"pel","muo")
```

To read it:
```python
cpobj = gmagcp.data.CrossPhase(20200101,"pel","muo")
```
where `cpobj` is an instance of `CrossPhase` which contains methods for 
plotting spectra and input data.