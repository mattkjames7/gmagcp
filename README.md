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

