import numpy as np
import xarray as xr

from ..constants import SQM_PER_HA, TC02_PER_TC, TC_PER_TBM, R


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
    years = xr.DataArray(np.arange(y0, y1 + 1), dims=("year",), name="year")
    tree_loss = xr.concat(
        [xr.where((ds["lossyear"] == year), 1.0, 0.0) for year in years], dim=years
    )

    return ds["agb"] * tree_loss * TC_PER_TBM * TC02_PER_TC


def compute_grid_area(da):
    """
    Compute the geographic area (in ha) of every pixel in the data array provided.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with spatial coordinates lon and lat

    Returns
    -------
    areacella : xarray.DataArray
        DataArray with grid cell areas in ha
    """

    dϕ = np.radians((da["lat"][1] - da["lat"][0]).values)
    dλ = np.radians((da["lon"][1] - da["lon"][0]).values)
    dA = R ** 2 * np.abs(dϕ * dλ) * np.cos(np.radians(da["lat"]))
    areacella = dA * xr.ones_like(da)

    return areacella / SQM_PER_HA


def coarsen_emissions(ds, mask_var='emissions', factor=100, method='sum'):
    """
    Coarsen emissions by the provided factor

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with emissions and spatial coordinates (lon and lat)
    factor : int
        The factor by for which the emissions data will be coarsened
    method: str
        Method used for coarsen, either "sum" or "mean"

    Returns
    -------
    ds : xarray.Dataset
        DataArray with grid cell areas in square meters
    """
    da_mask = ds[mask_var]
    if 'year' in ds.dims:
        da_mask = da_mask.isel(year=0, drop=True)

    da_area = compute_grid_area(da_mask)
    if method == 'sum':
        return (ds * da_area).coarsen(lat=factor, lon=factor).sum()
    elif method == 'mean':
        return (ds * da_area).coarsen(lat=factor, lon=factor).sum() / da_area.coarsen(
            lat=factor, lon=factor
        ).sum()
    else:
        raise (NotImplementedError('method must be either sum or mean'))
