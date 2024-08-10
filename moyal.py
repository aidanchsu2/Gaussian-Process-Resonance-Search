import numpy as np
import scipy
import uproot
import hist
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

import mplhep
mplhep.style.use('ROOT')

import matplotlib.pyplot as plt

def get_imd():
    """example of how to load a histogram from a file using uproot and hist

    will not work out of the box since the file path and the histogram name
    are not correct.
    """
    with uproot.open('path/to/file.root') as f:
        return f['imd-name'].to_hist()


def sim_imd():
    """mock-up a histogram with similar shape by sampling a Moyal distribution

    The `loc` (peak) and `scale` (width) of the Moyal was chosen just by eye
    so that both the rising and falling edges were within our normal mass range.
    """
    h = hist.Hist.new.Reg(250,0,2.5,label='Mass / GeV').Double()
    h.fill(scipy.stats.moyal.rvs(loc=0.5, scale=0.3, size=100_000))
    return h


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
    """

    x = histogram.axes[0].centers

    fit_selection = slice(None) # a.k.a. the "no slice" slice
    if blind_range is not None:
        if isinstance(blind_range, (tuple,list)) and len(blind_range)==2:
            fit_selection = (x < blind_range[0])|(x > blind_range[1])
        else:
            raise ValueError('blind_range is not a length-2 tuple defining the range of coordinates to blind')

    x_train = x[fit_selection]
    y_train = histogram.values()[fit_selection]
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


def plot(
    histogram: hist.Hist,
    gp: GaussianProcessRegressor,
    filename: str
):
    """Plot a comparison between the histogram and the (presumed already fit) GP

    Parameters
    ----------
    histogram: hist.Hist
        histogram to plot
    gp: GuassianProcessRegressor
        fit of histogram to plot as well
    filename: str
        path to file to save plot to
    """

    fig, (raw, pull) = plt.subplots(
        nrows = 2,
        height_ratios = [2, 1],
        sharex = 'col',
        gridspec_kw = dict(
            hspace = 0.05
        )
    )

    # hist has plotting methods already
    #   add label (for legend) and don't show the flow bins
    #   (default is to draw a little arrow hinting that something exists out there)
    histogram.plot(ax=raw, label='Histogram', flow=None)
    x = histogram.axes[0].centers
    mean_pred, std_pred = gp.predict(x.reshape(-1,1), return_std=True)
    art, = raw.plot(
        x, mean_pred,
        label='\n'.join([
            f'GP ({repr(gp.kernel_)})',
            'with 95% Confidence Interval'
        ])
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
    pull.set_xlabel(histogram.axes[0].label)
    mplhep.label.exp_label('HPS', llabel='Internal', rlabel='2016', ax=raw)

    pull_values = (histogram.values()-mean_pred)/np.sqrt(histogram.variances())
    pull.plot(x, pull_values)
    pull.set_ylabel(r'$(\mathrm{H} - \mathrm{GP})/\sigma_\mathrm{H}$')

    # draw blind range borders (if the fit was blinded)
    x_train = gp.X_train_[:,0]
    if len(x) != len(x_train):
        # there was a blind range
        size_blind_range = len(x)-len(x_train)
        # let's do some numpy shit to figure out what it was
        #
        # np.pad allows us to add some extra values on the end of x_train so that
        #        it can be the same shape as x
        # !=     checks the values of x vs the padded x_train to see which are equal
        # np.argmax returns the first index that is the "maximum" (for booleans True
        #           is bigger than False)
        blind_low = x[np.argmax(x != np.pad(x_train, (0, size_blind_range)))]
        # to get the upper edge, we do the same thing but with the arrays flipped
        blind_up = x[(
            len(x) - 1
            - np.argmax(np.flip(x != np.pad(x_train, (size_blind_range,0))))
        )]
        for ax in [raw, pull]:
            for edge in [blind_low, blind_up]:
                ax.axvline(edge, color='tab:red')

    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


def main():
    h = sim_imd()
    gp = fit(h, 1.*kernels.RBF(), blind_range=(0.9,1.1))
    # gp.kernel_ prints out the (now optimized) kernel
    # gp.X_train_ shows what the X training points were (helpful for confirming blind range)
    plot(h, gp, 'eg.png')


if __name__ == '__main__':
    main()
