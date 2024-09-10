import pickle
from pathlib import Path
from typing_extensions import Annotated
from typing import Tuple

import typer


from . import GaussianProcessModel
from . import kernels
from . import manipulation
from . import io
from . import mass_resolution


app = typer.Typer()

InputHistDefault = ('hps2016invMHisto10pc.root', 'invM_h')
InputHist = typer.Option(help = 'Input histogram file with the key to the histogram in that file.')


@app.command()
def fit_and_plot(
    name: Annotated[str, typer.Argument(help='name for this run, used as name for output files')],
    blind_range: Annotated[Tuple[float,float], typer.Option(help='edges of range to blind in GeV')] = None,
    search_mass: Annotated[float, typer.Option(help='search for this mass in GeV')] = None,
    low_lim : Annotated[float, typer.Option(help='lower limit of fit range in GeV')] = 0.033,
    up_lim : Annotated[float, typer.Option(help='upper limit of fit range in GeV')] = 0.179,
    rebin : Annotated[int,typer.Option(help='rebin factor to apply before fitting')] = 10,
    input : Annotated[Tuple[str,str], InputHist] = InputHistDefault,
    out_dir : Annotated[Path, typer.Option(help='output directory to write files to')] = Path.cwd()
):
    """Fit a GP model and plot the result a single time"""
    if blind_range is not None and search_mass is not None:
        print('You can only specify one of --blind-range or --search-mass.')
        typer.Exit(code=1)

    br = None
    if blind_range is not None:
        br = blind_range
    elif search_mass is not None:
        br = [
            search_mass - mass_resolution(search_mass),
            search_mass + mass_resolution(search_mass)
        ]
    gpm = GaussianProcessModel(
        h = io.load(*input),
        kernel = 1.0 * kernels.RBF() * kernels.DotProduct(),
        blind_range = br,
        modify_histogram = [
            manipulation.rebin_and_limit(rebin, low_lim, up_lim),
        ]
    )
    fig, axes = gpm.plot_comparison()
    filestem = out_dir / name
    fig.savefig(filestem.with_suffix('.png'), bbox_inches='tight')
    with open(filestem.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(gpm, f)


@app.command()
def inject_signal(
    output: Annotated[Path, typer.Argument(help='output file to write histogram to')],
    name: Annotated[str, typer.Argument(help='name of histogram in output file')],
    mass: Annotated[float, typer.Argument(help='inject signal of this mass in GeV')],
    nevents : Annotated[int, typer.Argument(help='number of events to inject')],
    mass_width: Annotated[float, typer.Argument(help='width of signal peak', show_default='determined by mass resolution')] = None,
    input : Annotated[Tuple[str,str], InputHist] = InputHistDefault
):
    """Inject signal into the IMD and then write the updated histogram out"""
    h = io.load(*input)
    mass_width = mass_resolution(mass) if mass_width is None else mass_width
    h = manipulation.inject_signal(
        location = mass,
        width = mass_width,
        amplitude = nevents
    )(h)
    io.write(output, name, h)


@app.command()
def search(
    output: Path,
    input : Annotated[Tuple[str,str], InputHist] = InputHistDefault
):
    """Search through mass points, performing a fit at each one"""
    for mass in tqdm(mass_range):
        gpm = GaussianProcessModel(
            h = input,
            kernel = 1.0*kernels.RBF()*kernels.DotProduct(),
            blind_range = (mass - 1.96*sigma_m, mass + 1.96*sigma_m),
            modify_histogram = [
                manipulation.rebin_and_limit(10, 0.033, 0.174)
            ]
        )


if __name__ == '__main__':
    app()
