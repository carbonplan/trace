import gc
import os
from datetime import datetime

import boto3
import dask
import fsspec
import geopandas as gpd
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


prep_training_dataset_delayed = dask.delayed(prep_training_dataset)


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
    biomass_variables = ['burned', 'treecover2000_mean', 'ecoregion'] + ['lat', 'lon']
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


def build_url(df):
    return "s3://carbonplan-climatetrace/v1/training/{}/{}/{}/data.parquet".format(
        df["PATH"], df["ROW"], df["year"]
    )


def add_parquet_urls(df):
    df = add_projection_info(df)
    df["year"] = df.apply(grab_year, axis=1)
    df["parquet_url"] = df.apply(build_url, axis=1)
    return df


def grab_all_scenes_in_tile(ul_lat, ul_lon):
    gdf = gpd.read_file(
        "https://prd-wret.s3-us-west-2.amazonaws.com/assets/"
        "palladium/production/s3fs-public/atoms/files/"
        "WRS2_descending_0.zip"
    )
    scenes_in_tile = gdf.cx[ul_lon : ul_lon + 10, ul_lat - 10 : ul_lat]

    return scenes_in_tile


def find_pertinent_scenes_for_shots(lat_tag, lon_tag, scenes_in_tile_gdf):

    file_mapper = fs.get_mapper(
        "carbonplan-climatetrace/v1/data/intermediates/biomass/{}_{}.zarr".format(lat_tag, lon_tag)
    )
    biomass = xr.open_zarr(file_mapper, consolidated=True).load().drop("spatial_ref")

    biomass_df = biomass.stack(unique_index=("record_index", "shot_number")).to_dataframe()

    biomass_df = biomass_df[biomass_df['biomass'].notnull()]

    # find all of the row/paths pertinent for the biomass shots
    biomass_gdf = gpd.GeoDataFrame(
        biomass_df, geometry=gpd.points_from_xy(biomass_df.lon, biomass_df.lat)
    ).set_crs("EPSG:4326")

    linked_gdf = gpd.sjoin(
        biomass_gdf,
        scenes_in_tile_gdf,  # gdf.cx[-125:-115,45:49], # gdf.cx[-ul_lon:-ul_lon+10,ul_lat-10:ul_lat],
        how="inner",
    )  # 'left' # by selecting inner you're grabbing the intersection (so dropping any shots
    # that don't have scenes or scenes that don't have shots)
    # what happens if a shot gets multiple scenes??? grab first? that's fine

    linked_gdf = add_parquet_urls(linked_gdf)

    all_parquet_files = linked_gdf["parquet_url"].unique()
    return all_parquet_files


def aggregate_parquet_files(
    lat_tag, lon_tag, all_parquet_files, write=True, access_key_id=None, secret_access_key=None
):

    full_df = None
    for url in all_parquet_files:
        df = pd.read_parquet(url)
        if full_df is not None:
            full_df = df
        else:
            full_df = pd.concat([full_df, df])
    if write:
        tile_parquet_file_path = fs.get_mapper(
            "carbonplan-climatetrace/v1/training/tiles/data_{}_{}.parquet".format(lat_tag, lon_tag)
        )
        utils.write_parquet(df, tile_parquet_file_path, access_key_id, secret_access_key)
    else:
        return full_df


def combine_parquet_files_full_tile(
    ul_lat, ul_lon, write=True, access_key_id=None, secret_access_key=None
):
    # grab all scenes in a given tile
    scenes_in_tile_gdf = grab_all_scenes_in_tile(ul_lat, ul_lon)

    lat_tag, lon_tag = utils.get_lat_lon_tags_from_bounding_box(ul_lat, ul_lon)

    all_parquet_files = find_pertinent_scenes_for_shots(lat_tag, lon_tag, scenes_in_tile_gdf)
    aggregate_parquet_files(
        lat_tag,
        lon_tag,
        all_parquet_files,
        write=write,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
