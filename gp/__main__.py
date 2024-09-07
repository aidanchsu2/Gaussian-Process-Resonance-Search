import pickle
from . import GaussianProcessModel, rebin_and_limit, kernels

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
        modify_histogram = [
            rebin_and_limit(rebin, low_lim, up_lim),
        ]
    )
    fig, axes = gpm.plot_comparison()
    if plot_filename is not None:
        fig.savefig(plot_filename, bbox_inches='tight')
    if display:
        plt.show()
    else:
        fig.clf()
    return gpm


def _cli_main():
    import argparse
    parser = argparse.ArgumentParser('python3 -m gp')
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
