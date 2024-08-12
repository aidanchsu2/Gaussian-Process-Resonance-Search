import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot
import hist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import mplhep
mplhep.style.use('ROOT')

import matplotlib.pyplot as plt


def sim_imd():
    """simulate the IMD by sampling from a moyal distribution

    The values of the two moyal distribution parameters were taken
    from a fit of a Moyal PDF to the 6.5% 2016 IMD. The returned
    histogram has the same binning as this IMD and roughly the same
    number of entries (~5M, some lost to overflow).

    Returns
    -------
    hist.Hist
        simulated IMD
    """
    import scipy
    return (
        hist.Hist.new
        .Reg(6000, 0.0, 0.3, label = 'Mass / GeV')
        .Double()
        .fill(
            scipy.stats.moyal.rvs(
                loc = 0.065,
                scale = 0.013,
                size = 5_000_000
            )
        )
    )


def load_imd(fp : str|Path, imd_name : str = 'invM_h'):
    """Load an IMD from the input file and make sure the x axis is labeled appropriately

    Parameters
    ----------
    filepath: str|Path
        path to ROOT file to load
    imd_name: str, optional, default invM_h
        key name of histogram in ROOT file

    Returns
    -------
    hist.Hist
        loaded IMD from the input file
    """
    with uproot.open(fp) as f:
        h = f[imd_name].to_hist()
        h.axes[0].label = 'Mass / GeV'
        return h


def _deduce_histogram(h: hist.Hist|str):
    """Deduce and return the histogram that should be used from the input specification

    Meant to be used within the construction of the GP model class below.

    Parameters
    ----------
    h: hist.Hist|str
        If a hist.Hist is given, use that as the histogram.
        If h is a str, there are two possible values.
        'sim' returns the result of sim_imd and 'real' returns the result of 'load_imd'
        with the filepath 'hps2016invMHisto10pc.root'.
        Any other str produces a ValueError and any other type produces a TypeError.

    Returns
    -------
    hist.Hist
        histogram following input specification
    """

    if isinstance(h, str):
        if h == 'sim':
            return sim_imd()
        elif h == 'real':
            # load histogram from ROOT file
            return load_imd('hps2016invMHisto10pc.root')
        else:
            raise ValueError(f'Histogram specification {h} not understood.')
    elif isinstance(h, hist.Hist):
        return h
    else:
        raise TypeError(f'Histogram specification of type {type(h)} not supported.')


def fit(
    histogram: hist.Hist,
    kernel,
    blind_range = None,
    **kwargs
) -> GaussianProcessRegressor:
    """fit the input histogram with a GP using the input kernel

    optionally, blind the fit to a range of the histogram

    Parameters
    ----------
    histogram: hist.Hist
        histogram to fit a GP to
    kernel:
        kernel to use in GP
    blind_range: 2-tuple, optional, default None
        range of histogram to "blind" the fit to
        (i.e. do /not/ use this range of values in fit)
    kwargs: dict[str,Any]
        all the rest of the keyword arguments are passed to GaussianProcessRegressor


    Note
    ----
    The running of this fit can take a while. The GP is manipulating N-dimensional
    matrices where N corresponds to the number of training points. Since there are
    6000 bins in the input invariant mass histogram, N=6000 which are really large
    matrices and can take a long time to run.
    The manipulation of large matrices is a problem _built_ for GPUs and hence
    we may want to switch to a GPU-possible implementation of GPs like GPyTorch[1].
    In the mean time, I would highly recommend pickling[2] the resulting fitted GP object
    so that the fit doesn't need to be re-run if you just want to make different plots
    with the predictions the GP makes.

        import pickle
        with open('helpful-name-for-this-gp-fit.pkl','wb') as f:
            pickle.dump(gp, f)

    and then somewhere else (e.g. in a notebook where you are playing with plots) you can

        import pickle
        with open('helpful-name-for-this-gp-fit.pkl','rb') as f:
            gp = pickle.load(f)


    [1]: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
    [2]: https://docs.python.org/3/library/pickle.html
    """

    x     = histogram.axes[0].centers
    value = histogram.values()

    fit_selection = (value > 0.0) # make sure something is in the bins
    if blind_range is not None:
        if isinstance(blind_range, (tuple,list)) and len(blind_range)==2:
            fit_selection = fit_selection&((x < blind_range[0])|(x > blind_range[1]))
        else:
            raise ValueError('blind_range is not a length-2 tuple defining the range of coordinates to blind')

    x_train = x[fit_selection]
    y_train = value[fit_selection]
    variance = histogram.variances()[fit_selection]

    if 'alpha' in kwargs:
        raise KeyError('alpha cannot be manually set. It is determined to be the variance of the histogram to fit')

    # update n_restarts_optimizer default to be 9 as is used in example
    kwargs.setdefault('n_restarts_optimizer', 9)
    _gp = GaussianProcessRegressor(
        kernel = kernel,
        alpha = variance,
        **kwargs
    )
    # fit expects a _column_ for x and a _row_ for y so we need to reshape x
    _gp.fit(x_train.reshape(-1,1), y_train)
    return _gp


class GaussianProcessModel:
    """Model the IMD with a Gaussian Process fit (GP)

    Parameters
    ----------
    h : hist.Hist|str
        specification of histogram
        can provide a hist.Hist or a string naming a specific one
    kernel :
        GP kernel to use for model
    blind_range: 2-tuple, optional, default None
        range of histogram to blind model to during fit
    modify_histogram: Callable
        function to modify histogram after it is loaded but before it is fitted
        could (for example) rebin, inject a signal bump, or limit the fitting range
    kwargs: dict[str,Any]
        rest of keyword arguments passed to the function fit
    """

    def __init__(
        self,
        h: hist.Hist|str,
        kernel, *,
        blind_range = None,
        modify_histogram = None,
        **kwargs
    ):
        _h = _deduce_histogram(h)
        self.histogram = _h if modify_histogram is None else modify_histogram(_h)
        self.blind_range = blind_range # store for plotting purposes
        self.model = fit(
            self.histogram,
            kernel,
            blind_range = self.blind_range,
            **kwargs
        )


    def plot(self):
        """Plot a comparison between the histogram and the (presumed already fit) GP"""
    
        fig, axes = plt.subplots(
            nrows = 3,
            height_ratios = [2, 1, 1],
            sharex = 'col',
            gridspec_kw = dict(
                hspace = 0.05
            ),
            figsize = (10,12)
        )

        raw, ratio, pull = axes

        # evaluate GP model prediction
        x = self.histogram.axes[0].centers
        mean_pred, std_pred = self.model.predict(x.reshape(-1,1), return_std=True)

        # RAW
        # hist has plotting methods already
        #   add label (for legend) and don't show the flow bins
        #   (default is to draw a little arrow hinting that something exists out there)
        self.histogram.plot(ax=raw, label='Observed Data', flow=None)
        art, = raw.plot(
            x, mean_pred,
            label= 'GP with 95% Confidence Interval'
        )
        raw.fill_between(
             x, mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred,
             alpha = 0.5, color = art.get_color()
        )
        raw.legend(
            title = f'Kernel: {repr(self.model.kernel_)}',
            title_fontsize = 'xx-small'
        )
        raw.set_ylabel('Event Count')
        # mpl default is to add some horizontal padding which I don't like
        raw.set_xlim(np.min(x), np.max(x))
        mplhep.label.exp_label('HPS', llabel='Internal', rlabel='2016 6.5%', ax=raw)

        combined_variance = self.histogram.variances()+std_pred**2
        positive_prediction = (mean_pred > 0)&(self.histogram.values() > 0)

        # RATIO
        ratio_values = (
            self.histogram.values()[positive_prediction]
            /mean_pred[positive_prediction]
        )
        ratio.plot(x[positive_prediction], ratio_values)
        ratio_err = ratio_values*np.sqrt(
            std_pred[positive_prediction]**2
            /mean_pred[positive_prediction]
        )
        ratio.fill_between(
            x[positive_prediction], ratio_values - ratio_err, ratio_values+ratio_err,
            alpha = 0.5
        )
        ratio.axhline(1, color='gray', ls=':')
        ratio.set_ylabel(r'Data / GP')

        # PULL
        pull_values = (
            (self.histogram.values()-mean_pred)[positive_prediction]
            /np.sqrt(combined_variance[positive_prediction])
        )
        pull.plot(x[positive_prediction], pull_values)
        pull.fill_between(
            x, np.full(x.shape, -2), np.full(x.shape, +2),
            color='gray', alpha=0.5
        )
        pull.set_ylabel(r'$(\mathrm{Data} - \mathrm{GP})/\sigma$')

        # FINAL CLEANUP

        for ax in axes[:-1]:
            ax.set_xlabel(None) # undo labeling to avoid ugliness
        axes[-1].set_xlabel(self.histogram.axes[0].label)
        
        if self.blind_range is not None:
            for e in self.blind_range:
                for ax in axes:
                    ax.axvline(e, color='tab:red')

        return fig, axes


@dataclass
class rebin_and_limit:
    """rebin an input histogram by the defined factor"""
    factor: int = None
    low_limit: float = None
    high_limit: float = None


    def __call__(self, h):
        return h[slice(
            hist.loc(self.low_limit) if self.low_limit is not None else None,
            hist.loc(self.high_limit) if self.high_limit is not None else None,
            hist.rebin(self.factor) if self.factor > 1 else None
        )]


def fit_and_plot(
    name,
    blind_range = None,
    low_lim = 0.033,
    up_lim = 0.179,
    rebin = 10,
    plot_filename = None,
    display = True
):
    gpm = GuassianProcessModel(
        h = 'real',
        kernel = 1.0 * kernels.RationalQuadratic(),
        blind_range = blind_range,
        modify_histogram = rebin_and_limit(rebin, low_lim, up_lim)
    )
    fig, axes = gpm.plot()
    if plot_filename is not None:
        fig.savefig(plot_filename, bbox_inches='tight')
    if display:
        plt.show()
    else:
        fig.clf()
    return gpm


def _cli_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='name for model pickle and image')
    parser.add_argument('--blind', nargs=2, type=float, help='bounds of region to blind')
    parser.add_argument('--rebin', type=int, help='rebin factor', default=10)
    parser.add_argument('--low-lim', type=float, default=0.033, help='lower limit of fit')
    parser.add_argument('--up-lim', type=float, default=0.179, help='upper limit of fit')
    args = parser.parse_args()

    gpm = fit_and_plot(
        args.name,
        blind_range = args.blind,
        rebin = args.rebin,
        low_lim = args.low_lim,
        up_lim = args.up_lim,
        plot_filename = f'{args.name}.png',
        display = False
    )
    with open(f'{args.name}.pkl','wb') as f:
        pickle.dump(gpm.model, f)

if __name__ == '__main__':
    _cli_main()
