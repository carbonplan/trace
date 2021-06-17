import gc
import os
from datetime import datetime

import boto3
import dask
import fsspec
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray  # for the extension to load
import utm
import xarray as xr
from pyproj import CRS
from rasterio.session import AWSSession
from s3fs import S3FileSystem

from carbonplan_trace.v1 import load, utils
from carbonplan_trace.v1.inference import reproject_dataset_to_fourthousandth_grid
from carbonplan_trace.v1.landsat_preprocess import scene_seasonal_average

# flake8: noqa

fs = S3FileSystem(profile='default', requester_pays=True)


def add_aster_worldclim(data, tiles, lat_lon_box=None):
    data = load.aster(data, tiles, lat_lon_box=lat_lon_box)
    data = load.worldclim(data)
    return data


def prep_training_dataset(
    path,
    row,
    year,
    access_key_id,
    secret_access_key,
    training_write_bucket=None,
    bands_of_interest='all',
    season='JJA',
):
    core_session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name='us-west-2',
    )
    aws_session = AWSSession(core_session, requester_pays=True)
    fs = S3FileSystem(key=access_key_id, secret=secret_access_key, requester_pays=True)
    with dask.config.set(
        scheduler='single-threaded'
    ):  # this? **** #threads #single-threaded # threads??
        with rio.Env(aws_session):

            # create the landsat scene for that year
            landsat_ds = scene_seasonal_average(
                path,
                row,
                year,
                access_key_id,
                secret_access_key,
                aws_session,
                core_session,
                fs,
                write_bucket='s3://carbonplan-climatetrace/v1/',
                bands_of_interest='all',
                season=season,
                landsat_generation='landsat-7',
            )
            # add in other datasets
            landsat_zone = landsat_ds.utm_zone_number + landsat_ds.utm_zone_letter
            data, tiles, bounding_box = reproject_dataset_to_fourthousandth_grid(
                landsat_ds, zone=landsat_zone
            )
            del landsat_ds
            data = add_aster_worldclim(data, tiles, lat_lon_box=bounding_box).load()
            # here we take it to dataframe
            df = create_combined_landsat_biomass_df(data, tiles, year)
            del data
            utils.write_parquet(df, training_write_bucket, access_key_id, secret_access_key)


def create_combined_landsat_biomass_df(data, tiles, year):
    '''
    Add landsat info for each biomass entry

    Parameters
    ----------
    data : xarray dataset
        landsat dataset to extract information from
    biomass_df : pandas dataframe
        dataframe with biomass, ancillary variables, and lat lon for linking to
        landsat
    year : pandas dataframe
        the year from the landsat scene that you want to grab from biomass_df
    Returns
    -------
    df : pandas dataframe
        The df with the biomass (and potentially other variables) and
        the corresponding landsat data
    '''
    biomass_variables = (
        [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
        + ['burned', 'treecover2000_mean']
        + ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']
        + ['elev', 'slope', 'aspect', 'lat', 'lon']
    )
    # open all the biomass tiles
    # don't need to do the bounding box trimming because
    # it isn't spatial data (it's df)
    biomass_df = load.biomass(tiles, year)[biomass_variables]
    # TODO CHECK THIS LINE!!
    df = (
        data.sel(
            x=xr.DataArray(biomass_df['lon'].values, dims='shot'),
            y=xr.DataArray(biomass_df['lat'].values, dims='shot'),
            method='nearest',
        )
        .to_dataframe()
        .drop(['spatial_ref'], axis=1)
    )
    del data
    df.index = biomass_df.index
    df = pd.concat([biomass_df, df], axis=1)

    # remove any values that got masked - we might want to do some tolerance here
    df = df.dropna(how='any')
    return df
