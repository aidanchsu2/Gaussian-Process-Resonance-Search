# Gaussian-Process-Resonance-Search

Using Gaussian Processes to model the Invariant Mass Distribution (IMD) within HPS.

Originally developed this with Python 3.10.12 and the requirements listed
in [requirements.txt](requirements.txt).

## Set Up
```
python3 -m venv .venv --prompt hps-gp
. .venv/bin/activate
pip install -r requirements.txt
# download IMD ROOT file from somewhere
```

## Start Working
```
. .venv/bin/activate
```

## Run

### From the command line
```
python gp.py noblind-eg
```

### In an notebook
Below are a few example cells with the output omitted for brevity.
In a real notebook, many of these cells end with statements that
return something and so those values/figures/reprs will be shown.

```python
# have modules be re-loaded when cells are executed
# this is helpful for us since then we can edit gp.py
# without needing to manually restart the kernel
%load_ext autoreload
%autoreload 2
```

```python
from gp import GaussianProcessModel, kernels, rebin_and_limit, load_imd
import pickle
import matplotlib.pyplot as plt
```

```python
m = GaussianProcessModel(
    h = 'real', # use real IMD from ROOT file in this directory
    kernel = 1.0 * kernels.RationalQuadratic(), # provide kernel to use
    blind_range = (0.045, 0.055),
    modify_histogram = rebin_and_limit(10, 0.033, 0.179)
)
m.model
```

```python
m.model.kernel_
```

```python
fig, axes = m.plot()
plt.show()
```
