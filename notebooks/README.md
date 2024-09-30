# Example Notebooks
These are examples of using `gp` within a notebook.
Since `gp` is a module that is not installed, the easiest
way to make it available to your notebook is to symlink
the source code to the same directory as your notebook.

We also have a common ROOT file holding the IMD that we
use for testing and this can be symlinked to the same location
so that it is also available.
```
cd path/to/notebook
ln -s path/to/gp .
ln -s path/to/hps2016invMHisto10pc.root .
```

For example, we can enable using notebooks in this directory with
```
cd notebooks
ln -s ../gp .
ln -s ../hps2016invMHisto10pc.root .
```
