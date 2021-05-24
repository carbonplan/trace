#!/usr/bin/env python

import dask
import fsspec
import numcodecs
import numpy as np
import rasterio
import xarray as xr
import zarr
from tqdm import tqdm

from carbonplan_trace.v0.data import cat

TC02_PER_TC = 3.67
TC_PER_TBM = 0.5
SQM_PER_HA = 10000
ORNL_SCALING = 0.1


def _preprocess(da, lat=None, lon=None):
    '''
    Adjust names of lat/lon
    to conform with that required by alter functions

    Parameters
    ----------
    da : xarray data array
        Hansen data array
    lat : float <--- don't know if this is correct
        The latitude of northwest corner of the region selected
    lon : float <--- don't know if this is correct
        The longitude of northwest corner of the region selected

    Returns
    -------
    da: xarray data array
        A nicely formatted data array for use down the line.
    """

    '''
    da = da.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
    if lat is not None:
        da = da.assign_coords(lat=lat, lon=lon)
    return da


def calc_emissions(ds):
    """
    Multiply above ground biomass by change in treecover
    (a timeseries), then scale to mass of carbon and
    finally to mass of CO2.

    Parameters
    ----------
    ds : xarray dataset
        Dataset including aboveground biomass and timeseries of
        change in treecover.

    Returns
    -------
    emissions : xarray data array
        Timeseries (for the full record) of emissions due to
        forest tree cover disturbance.
    """

    d_biomass = ds["agb"] * ds["d_treecover"]
    emissions = d_biomass * TC_PER_TBM * TC02_PER_TC
    return emissions


def calc_one_tile(ds):
    """
    Multiply above ground biomass by change in treecover
    (a timeseries), then scale to mass of carbon and
    finally to mass of CO2.

    Parameters
    ----------
    ds : xarray dataset
        Dataset including aboveground biomass and timeseries of
        change in treecover.

    Returns
    -------
    emissions : xarray data array
        Timeseries (for the full record) of emissions due to
        forest tree cover disturbance.
    """
    years = xr.DataArray(range(2001, 2021), dims=("year",), name="year")
    loss_frac = []
    for year in years:
        loss_frac.append(xr.where((ds["lossyear"] == year), ds["treecover2000"], 0))
    ds["d_treecover"] = xr.concat(loss_frac, dim=years)
    ds["emissions"] = calc_emissions(ds)
    return ds


def open_hansen_change_tile(lat, lon, emissions=False):

    """
    Open single tile from the Hansen 2020 dataset and then
    massage it into a format for use by the rest of the routines.

    Parameters
    ----------
    lat : float
        The latitude of the northwest corner
    lon : float
        The longitude of the northwest corner of the
    Returns
    -------
    ds : xarray dataset
        Dataset containing tree cover, above ground biomass,
        loss year. Will be a timeseries spanning from 2001 to 2020.
    """

    ds = xr.Dataset()

    # Min global forest change data
    variables = ["treecover2000", "gain", "lossyear", "datamask"]  # , "first", "last"]
    for v in variables:
        da = cat.hansen_change(variable=v, lat=lat, lon=lon).to_dask().pipe(_preprocess)
        # force coords to be identical
        if ds:
            da = da.assign_coords(lat=ds.lat, lon=ds.lon)
        ds[v] = da

    ds["treecover2000"] /= 100.0
    ds["lossyear"] += 2000

    # Hansen biomass
    ds["agb"] = (
        cat.gfw_biomass(lat=lat, lon=lon).to_dask().pipe(_preprocess, lat=ds.lat, lon=ds.lon)
    )
    if emissions:
        # Hansen emissions
        ds["emissions_ha"] = (
            cat.hansen_emissions_ha(lat=lat, lon=lon)
            .to_dask()
            .pipe(_preprocess, lat=ds.lat, lon=ds.lon)
        )
        ds["emissions_px"] = (
            cat.hansen_emissions_px(lat=lat, lon=lon)
            .to_dask()
            .pipe(_preprocess, lat=ds.lat, lon=ds.lon)
        )

    return ds


# @dask.delayed
def process_one_tile(lat, lon):

    """
    Given lat and lon to select a region, calculate the
    corresponding emissions for each year from 2001 to 2020

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees

    Returns
    -------
    url : string
        Url where a processed tile is located
    """

    url = f"gs://carbonplan-climatetrace/v0.1/tiles/{lat}_{lon}.zarr"

    encoding = {"emissions": {"compressor": numcodecs.Blosc()}}

    mapper = fsspec.get_mapper(url)

    with dask.config.set(scheduler="threads"):
        ds = open_hansen_change_tile(lat, lon)
        ds = calc_one_tile(ds)[["emissions"]]
        ds = ds.chunk({"lat": 4000, "lon": 4000, "year": 2})
        ds.to_zarr(mapper, encoding=encoding, mode="w", consolidated=True)
        return url


def main():
    # start the cluster where you'll run everything
    #     gateway = Gateway()
    #     options = gateway.cluster_options()
    #     options.worker_cores = 2
    #     options.worker_memory = 24
    #     cluster = gateway.new_cluster(cluster_options=options)
    #     cluster.adapt(minimum=1, maximum=300)

    #     client = cluster.get_client()
    with fsspec.open(
        "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/treecover2000.txt"
    ) as f:
        lines = f.read().decode().splitlines()
    print("We are working with {} different files".format(len(lines)))

    # keys for all global tiles - this makes
    # a big box across the globe and the processes below will just skip the cells
    # that don't have valid data
    lat_tags = [
        "80N",
        "70N",
        "60N",
        "50N",
        "40N",
        "30N",
        "20N",
        "10N",
        "00N",
        "10S",
        "20S",
        "30S",
        "40S",
        "50S",
    ]
    lon_tags = [f"{n:03}W" for n in np.arange(10, 190, 10)] + [
        f"{n:03}E" for n in np.arange(0, 190, 10)
    ]

    # the arrays where you'll throw your active lat/lon permutations
    lats = []
    lons = []
    for line in lines:
        pieces = line.split("_")
        lat = pieces[-2]
        lon = pieces[-1].split(".")[0]

        if (lat in lat_tags) and (lon in lon_tags):
            lats.append(lat)
            lons.append(lon)

    # Compute emissions for every lat/lon that you've selected
    tiles = []
    for lat, lon in tqdm(list(zip(lats, lons))):
        try:
            print(lat, lon)
            tiles.append(process_one_tile(lat, lon))
        except (rasterio.errors.RasterioIOError, zarr.errors.PathNotFoundError):
            print("no tile available at {} {}".format(lat, lon))


if __name__ == "__main__":
    main()
