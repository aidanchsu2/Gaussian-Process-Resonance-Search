# Gaussian-Process-Resonance-Search

Using Gaussian Processes to model the Invariant Mass Distribution (IMD) within HPS.

Originally developed this with Python 3.10.12 and the requirements listed
in [requirements.txt](requirements.txt). More detail about implementation
focused on a new-developer audience is in [NOTES.md](NOTES.md).

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
python -m gp --help
```
There are many sub-commands each with their own options.
The `typer` package is used to provide a pretty-printed and detailed
help message that you are encouraged to explore.

### In an notebook
The [notebooks](notebooks) directory contains some example notebooks
of using `gp`.

> [!WARNING]
> While notebooks are very helpful for trying out new things and
> editing plots, they are difficult to track and keep reproducible.
> For this reason, you should only use notebooks as a "playground"
> and move anything that is working into the python code of the module
> itself to be usable by others.
