"""Different functions useful for manipulating histograms in memory"""

import hist


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
        h[:] = h.values() + signal_samples
        return h