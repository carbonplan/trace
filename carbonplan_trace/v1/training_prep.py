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
from carbonplan_trace.v1.glas_allometric_eq import ECO_TO_REALM_MAP
from carbonplan_trace.v1.inference import reproject_dataset_to_fourthousandth_grid
from carbonplan_trace.v1.landsat_preprocess import scene_seasonal_average

# flake8: noqa

fs = S3FileSystem()


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
    error='raise',
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
            try:
                landsat_ds = scene_seasonal_average(
                    path=path,
                    row=row,
                    year=year,
                    access_key_id=access_key_id,
                    secret_access_key=secret_access_key,
                    aws_session=aws_session,
                    core_session=core_session,
                    fs=fs,
                    bands_of_interest=bands_of_interest,
                    landsat_generation='landsat-7',
                )
                if landsat_ds:
                    # reproject from utm to lat/lon
                    landsat_zone = landsat_ds.utm_zone_number + landsat_ds.utm_zone_letter
                    # sets null value to np.nan
                    write_nodata(landsat_ds)
                    data, tiles, bounding_box = reproject_dataset_to_fourthousandth_grid(
                        landsat_ds, zone=landsat_zone
                    )
                    del landsat_ds
                    # add in other datasets
                    data = add_aster_worldclim(data, tiles, lat_lon_box=bounding_box).load()
                    # here we take it to dataframe and add in realm information
                    df = create_combined_landsat_biomass_df(data, tiles, year, bounding_box)
                    del data
                else:
                    df = pd.DataFrame({})
                print('length of data', len(df))

                # according to realm, save to the correct bucket
                if len(df) == 0:
                    output_filepath = (
                        f'{training_write_bucket}/no_data/{year}/{path:03d}{row:03d}.parquet'
                    )
                    print(output_filepath)
                    mock = pd.DataFrame({'no_data': [1]})
                    utils.write_parquet(mock, output_filepath, access_key_id, secret_access_key)
                else:
                    for realm, sub in df.groupby('realm'):
                        output_filepath = (
                            f'{training_write_bucket}/{realm}/{year}/{path:03d}{row:03d}.parquet'
                        )
                        print(output_filepath)
                        utils.write_parquet(sub, output_filepath, access_key_id, secret_access_key)

                return ('pass', f'{year}/{path:03d}{row:03d}')

            except Exception as e:
                if error == 'raise':
                    raise e
                else:
                    return ('error', f'{year}/{path:03d}{row:03d}', e)


prep_training_dataset_delayed = dask.delayed(prep_training_dataset)


def create_combined_landsat_biomass_df(data, tiles, year, bounding_box):
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
    bounding_box: list
        [min_lat, max_lat, min_lon, max_lon]
    Returns
    -------
    df : pandas dataframe
        The df with the biomass (and potentially other variables) and
        the corresponding landsat data
    '''
    biomass_variables = ['biomass', 'burned', 'treecover2000_mean', 'ecoregion', 'lat', 'lon']
    # open all the biomass tiles
    # don't need to do the bounding box trimming because
    # it isn't spatial data (it's df)
    biomass_df = load.biomass(tiles, year)[biomass_variables]
    min_lat, max_lat, min_lon, max_lon = bounding_box

    biomass_df = biomass_df.loc[
        (biomass_df.lat >= min_lat)
        & (biomass_df.lat <= max_lat)
        & (biomass_df.lon >= min_lon)
        & (biomass_df.lon <= max_lon)
    ]
    biomass_df['realm'] = biomass_df.ecoregion.apply(ECO_TO_REALM_MAP.__getitem__)
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
