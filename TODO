-[x] re-organize code to be split across multiple files into a manageable form
-[x] signal injector-inator optionally write to file
-[x] mass resolution functions
-[x] auto-cacluation of blind range
-[x] search scan through mass points
-[ ] p-value of excess at mass point
-[ ] 95% CL maximum at mass point

Modules

gp
  - class GaussianProcessModel
    - main API with construction, evaluation, and plotting
  - __main__ : run when gp module is executed as a program
  - mod _fit : fitting implementations
  - mod _plot : plotting utilities
  - mod _hist : histograms utilities
    - mod manipulation : modify histograms in a known way
      - class rebin_and_limit : rebin and restrict the fit range
      - class signal_inject : add bump on top of the signal
    - mod io : input and output of histograms from/to files
      - def read : read in a histogram from a file
      - def write : write out histogram to a file
  - mod _mass_resolution : mass resolution definition
