import pickle
from dataclasses import dataclass

import numpy as np
import uproot
import hist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import mplhep
mplhep.style.use('ROOT')

import matplotlib.pyplot as plt


def _deduce_histogram(h: hist.Hist|str):
    if isinstance(h, str):
        if h == 'sim':
            # simulate a IMD with a moyal distribution
            # TODO: update to be closer to realistic
            import scipy
            return (
                hist.Hist.new.Reg(250,0,2.5,label='Mass / GeV').Double()
                .fill(scipy.stats.moyal.rvs(loc=0.5, scale=0.3, size=1_000_000))
            )
        elif h == 'real':
            # load histogram from ROOT file
            with uproot.open('hps2016invMHisto10pc.root') as f:
                h = f['invM_h'].to_hist()
                h.axes[0].label = 'Mass / GeV' # update label to make later plots easier
                return h
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
        range of histogram to "blind" the fit to (i.e. do /not/ use this range of values in fit)
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
        function to modify histogram after it is loaded
        could for example rebin or inject a signal bump
    kwargs: dict[str,Any]
        rest of keyword arguments passed to fit
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
    
        fig, (raw, pull) = plt.subplots(
            nrows = 2,
            height_ratios = [2,1],
            sharex = 'col',
            gridspec_kw = dict(
                hspace = 0.05
            )
        )
    
        # hist has plotting methods already
        #   add label (for legend) and don't show the flow bins
        #   (default is to draw a little arrow hinting that something exists out there)
        self.histogram.plot(ax=raw, label='Histogram', flow=None)
        x = self.histogram.axes[0].centers
        mean_pred, std_pred = self.model.predict(x.reshape(-1,1), return_std=True)
        art, = raw.plot(
            x, mean_pred,
            #label='\n'.join([
             #   f'GP ({repr(self.model.kernel_)})',
              #  'with 95% Confidence Interval'
            #])
            
            label='GP'
        )
        raw.fill_between(
             x, mean_pred - 1.96*std_pred, mean_pred + 1.96*std_pred,
             alpha = 0.5, color = art.get_color()
        )
        raw.legend()
        raw.set_ylabel('Event Count')
        # mpl default is to add some horizontal padding which I don't like
        raw.set_xlim(np.min(x), np.max(x))
        raw.set_xlabel(None) # undo labeling to avoid ugliness
        pull.set_xlabel(self.histogram.axes[0].label)
        mplhep.label.exp_label('HPS', llabel='Internal', rlabel='2016', ax=raw)
    
        sl = (self.histogram.values() > 0)
        pull_values = (self.histogram.values()-mean_pred)[sl]/np.sqrt(self.histogram.variances()[sl])
        pull.plot(x[sl], pull_values)
        pull.fill_between(
            x, np.full(x.shape, -2), np.full(x.shape, +2),
            color='gray', alpha=0.5
        )
        pull.set_ylabel(r'$(\mathrm{H} - \mathrm{GP})/\sigma_\mathrm{H}$')

        if self.blind_range is not None:
            for e in self.blind_range:
                for ax in [raw, pull]:
                    ax.axvline(e, color='tab:red')

        return fig, (raw, pull)
        
        
    def plot_pull_histogram(self):
        x = self.histogram.axes[0].centers
        mean_pred, std_pred = self.model.predict(x.reshape(-1,1), return_std=True)
        sl = (self.histogram.values() > 0)

        pull_values = (self.histogram.values()-mean_pred)[sl]/np.sqrt(self.histogram.variances()[sl])
        pp = plt.hist(pull_values,40,(-10,10),label=r'Pull $(H - GP)/\sigma$')
        plt.xlabel(r'Pull $(H - GP)/\sigma$')
        plt.ylabel('Frequency (# of bins)')
        
        return pp

    def plot_raw(self):
        self.histogram.plot(label='Histogram', flow=None)
        plt.ylabel('Event Count')

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


@dataclass
class inject_signal:
    """add a signal bump onto the histogram

    Parameters
    ----------
    amplitude: float
        total number of events to have bump represent
    width: float
        width of gaussian signal bump (standard deviation)
    location: float
        center of gaussian signal bump (mean)
    """
    amplitude: float
    width: float
    location: float
    

    def __call__(self, h):
        """UNTESTED"""
        import scipy
        signal_samples = self.amplitude*scipy.stats.norm.pdf(
            h.axes[0].centers,
            loc=self.location,
            scale=self.width
        )
        h[:] += signal_samples
        return h


def fit_and_plot(
    name,
    blind_range = None,
    low_lim = 0.033,
    up_lim = 0.179,
    rebin = 10,
    plot_filename = None,
    display = True
):
    gpm = GaussianProcessModel(
        h = 'real',
        kernel = 1.0 * kernels.RBF() * kernels.DotProduct(),
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

    gpm = GaussianProcessModel(
        h = 'real',
        kernel = 1.0 * kernels.RationalQuadratic(),
        blind_range = args.blind,
        modify_histogram = rebin_and_limit(args.rebin, args.low_lim, args.up_lim)
    )
    with open(f'{args.name}.pkl','wb') as f:
        pickle.dump(gpm.model, f)
    fig, axes = gpm.plot()
    fig.savefig(f'{args.name}.png', bbox_inches='tight')
    fig.clf()


if __name__ == '__main__':
    _cli_main()
