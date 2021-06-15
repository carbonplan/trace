import xarray as xr
from tenacity import retry

from . import cat

dtypes = {"treecover2000": "float32", "lossyear": "int16", "agb": "float32"}


def _preprocess(da, lat=None, lon=None):
    '''
    Adjust names of lat/lon
    to conform with that required by alter functions

    Parameters
    ----------
    da : xarray.DataArray
        Hansen data array
    lat : array_like
        The latitude of northwest corner of the region selected
    lon : array_like
        The longitude of northwest corner of the region selected

    Returns
    -------
    da: xarray.DataArray
        A nicely formatted data array for use down the line.
    """

    '''
    da = da.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
    if lat is not None:
        da = da.assign_coords(lat=lat, lon=lon)
    return da


@retry
def open_hansen_change_tile(lat, lon):
    """
    Open single tile from the Hansen 2020 dataset and then
    massage it into a format for use by the rest of the routines.

    Parameters
    ----------
    lat : float
        The latitude of the northwest corner of the tile
    lon : float
        The longitude of the northwest corner of the tile

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing tree cover, above ground biomass,
        loss year. Will be a timeseries spanning from 2001 to 2020.
    """

    ds = xr.Dataset()

    # Global forest change data
    variables = ["treecover2000", "lossyear"]
    for v in variables:
        da = cat.hansen_change(variable=v, lat=lat, lon=lon).to_dask().pipe(_preprocess)
        da = da.astype(dtypes[v])
        # force coords to be identical
        if ds:
            da = da.assign_coords(lat=ds.lat, lon=ds.lon)
        ds[v] = da

    ds["treecover2000"] /= 100.0
    ds["lossyear"] += 2000

    # Hansen biomass
    ds["agb"] = (
        cat.gfw_biomass(lat=lat, lon=lon)
        .to_dask()
        .pipe(_preprocess, lat=ds.lat, lon=ds.lon)
        .astype(dtypes["agb"])
    )

    return ds
