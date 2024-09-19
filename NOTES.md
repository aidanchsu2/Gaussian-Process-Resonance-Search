### TODO
-[x] re-organize code to be split across multiple files into a manageable form
-[x] signal injector-inator optionally write to file
-[x] mass resolution functions
-[x] auto-cacluation of blind range
-[x] search scan through mass points
-[ ] p-value of excess at mass point
-[ ] 95% CL maximum at mass point

# Extra Implementation Notes

## Modules
The parent module `gp` holds several sub-modules that are
organized in a (hopefully) logical way.

`gp`
  - `class GaussianProcessModel` : main API with construction, evaluation, and plotting
  - `__main__` : run when gp module is executed as a program
  - `_fit.py` : fitting implementations, currently only a single implementation via scikit-learn
  - `_plot.py` : plotting utilities
  - `_hist/` : histograms utilities
    - `manipulation.py` : modify histograms in a known way
      - `class rebin_and_limit` : rebin and restrict the fit range
      - `class signal_inject` : add bump on top of the signal
    - `io.py` : input and output of histograms from/to files
      - `def read` : read in a histogram from a file
      - `def write` : write out histogram to a file
  - `_mass_resolution.py` : mass resolution definition

Python3 converts directory hierarchies into module hierarchies, so
`gp/_hist/manipulation.py` gets converted into `gp._hist.manipulation`
within python. This is helpful for us so we can keep files small and focused
on specific tasks.
Whenever Python3 enters a new directory while loading modules, it looks for
a `__init__.py` file to load before continuing. This offers a helpful point
for us to rename things in a helpful manner or define shared behavior.
`gp/__init__.py` is a basic example of this where we rename some submodules
that may be helpful and we define the shared API class.
