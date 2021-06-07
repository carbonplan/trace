import xarray as xr

from ..constants import TC02_PER_TC, TC_PER_TBM


def calc_emissions(ds, y0=2001, y1=2020):
    """
    Multiply above ground biomass by change in treecover
    (a timeseries), then scale to mass of carbon and
    finally to mass of CO2.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset including aboveground biomass and timeseries of change in treecover.

    Returns
    -------
    emissions : xarray.DataArray
        Timeseries (for the full record) of emissions due to forest tree cover disturbance.
    """
    years = xr.DataArray(range(y0, y1 + 1), dims=("year",), name="year")
    loss_frac = []
    for year in years:
        loss_frac.append(xr.where((ds["lossyear"] == year), ds["treecover2000"], 0))
    ds["d_treecover"] = xr.concat(loss_frac, dim=years)

    return ds["agb"] * ds["d_treecover"] * TC_PER_TBM * TC02_PER_TC
