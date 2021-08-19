import gc
import os
import time

import boto3
import dask
import fsspec
import joblib
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray
import utm
import xarray as xr
import xgboost as xgb
from pyproj import CRS
from rasterio.session import AWSSession
from s3fs import S3FileSystem

import carbonplan_trace.v1.model as m
from carbonplan_trace.v1.glas_allometric_eq import ECO_TO_REALM_MAP
from carbonplan_trace.v1.landsat_preprocess import scene_seasonal_average

from ..v1 import load, utils

# flake8: noqa

fs = S3FileSystem(requester_pays=True)


def write_nodata(ds):
    for var in ds.data_vars:
        ds[var].rio.write_nodata(np.nan, inplace=True)


def write_crs_dataset(ds, zone=None, overwrite=False):
    '''
    This function will set a CRS for a dataset (whether or not
    one already exists!) so be sure you want to do that!
    '''
    if zone is None:
        zone = '{}{}'.format(ds.utm_zone_number, ds.utm_zone_letter)
    crs = CRS.from_dict({'proj': 'utm', 'zone': zone})
    ds = ds.rio.set_crs(crs)
    return ds


def check_mins_maxes(ds):
    lat_lon_crs = CRS.from_epsg(4326)
    reprojected = ds.rio.reproject(lat_lon_crs)
    min_lat = reprojected.y.min().values
    max_lat = reprojected.y.max().values
    min_lon = reprojected.x.min().values
    max_lon = reprojected.x.max().values
    return min_lat, max_lat, min_lon, max_lon


def create_target_grid(min_lat, max_lat, min_lon, max_lon):
    tiles = utils.find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon)
    if len(tiles) > 0:
        full_target_ds = utils.open_and_combine_lat_lon_data(
            's3://carbonplan-climatetrace/intermediate/ecoregions_mask/',
            tiles=tiles,
            lat_lon_box=[min_lat, max_lat, min_lon, max_lon],
            consolidated=False,
        )
        full_target_ds = full_target_ds.rename({'lat': 'y', 'lon': 'x'})
        buffer = 0.01
        target = full_target_ds.sel(
            y=slice(min_lat - buffer, max_lat + buffer), x=slice(min_lon - buffer, max_lon + buffer)
        )
        target.attrs["crs"] = "EPSG:4326"
        del full_target_ds
        return target, tiles
    else:
        return None, []


def reproject_dataset_to_fourthousandth_grid(ds, zone=None):
    ds = write_crs_dataset(ds, zone=zone)
    min_lat, max_lat, min_lon, max_lon = check_mins_maxes(ds)
    if max_lon >= 170 and min_lon <= -170:
        data_list, tiles_list, bounding_box_list = [], [], []

        target_east, tiles_east = create_target_grid(min_lat, max_lat, max_lon, 180)
        if len(tiles_east) > 0:
            reprojected_east = ds.rio.reproject_match(target_east).compute()
            reprojected_east = reprojected_east.where(reprojected_east < 1e100)
            data_list.append(reprojected_east)
            tiles_list.append(tiles_east)
            bounding_box_list.append([min_lat, max_lat, max_lon, 180])
        del target_east

        target_west, tiles_west = create_target_grid(min_lat, max_lat, -180, min_lon)
        if len(tiles_west) > 0:
            reprojected_west = ds.rio.reproject_match(target_west).compute()
            reprojected_west = reprojected_west.where(reprojected_west < 1e100)
            data_list.append(reprojected_west)
            tiles_list.append(tiles_west)
            bounding_box_list.append([min_lat, max_lat, -180, min_lon])
        del target_west

        return data_list, tiles_list, bounding_box_list

    else:
        target, tiles = create_target_grid(min_lat, max_lat, min_lon, max_lon)
        # the numbers aren't too big but if we normalize they might turn into decimals
        reprojected = ds.rio.reproject_match(target).compute()
        del ds
        del target
        reprojected = reprojected.where(reprojected < 1e100)
        return [reprojected], [tiles], [[min_lat, max_lat, min_lon, max_lon]]


def dataset_to_tabular(ds):
    '''
    Convert dataset to tabular form for inference

    Parameters
    ----------
    ds : xarray dataset
        xarray dataset with multiple bands

    Returns
    -------
    df : pandas dataframe
        dataframe with columns of bands

    '''
    df = ds.to_dataframe()
    del ds
    # drop any nan values so we only carry around pixels we have landsat for
    # this will drop both the parts of the dataset that are empty because
    # the landsat scenes might be rotated w.r.t. the x/y grid
    # but will also drop any cloud-masked regions
    # TODO: further investigate %-age of nulls and root cause
    df = df.dropna(how='any').reset_index()
    return df


def convert_to_lat_lon(df, utm_zone_number, utm_zone_letter):
    '''
    Given dataframe with x/y coordinates, project
    into the correct lat/lon coordinates, based upon UTM zone.

    Parameters
    ----------
    df : pandas dataframe
        geodataframe whose rows are entries for each row/path scene. Must
        include variables 'lon', 'lat'
    utm_zone_number : str/int
        string or int for the zone number (longitude) appropriate for that
        scene as defined by USGS
    utm_zone_letter : str
        string or int for the zone letter (latitude) appropriate for that
        scene as defined by USGS
    Returns
    -------
    df : pandas dataframe
        The projected information for each pixel
    '''

    return utm.to_latlon(df['x'], df['y'], int(utm_zone_number), utm_zone_letter)


def add_all_variables(data, tiles, year, lat_lon_box=None):
    data = load.aster(data, tiles, lat_lon_box=lat_lon_box)
    data = load.worldclim(data)
    data = load.treecover2000(data, tiles)
    data = load.ecoregion(data, tiles, lat_lon_box=lat_lon_box)

    return data


def make_inference(input_data, model):
    """
    input_data is assumed to be a pandas dataframe, and model uses standard sklearn API with .predict
    """
    input_data['NIR_V'] = m.calc_NIR_V(input_data)
    input_data = input_data.replace([np.nan, np.inf, -np.inf, None], np.nan)
    input_data = input_data.dropna(subset=m.features)
    gc.collect()
    print(f'predicting on {len(input_data)} records')
    t0 = time.time()
    with joblib.parallel_backend('threading', n_jobs=8):
        model.n_jobs = 8
        input_data['biomass'] = model.predict(input_data)
    t1 = time.time()
    print(f'took {round(t1-t0)} seconds')
    return input_data[['x', 'y', 'biomass']]


def predict(
    model_folder,
    path,
    row,
    year,
    access_key_id,
    secret_access_key,
    output_write_bucket=None,
    input_write_bucket=None,
    bands_of_interest='all',
):
    core_session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name='us-west-2',
    )
    aws_session = AWSSession(core_session, requester_pays=True)
    fs = S3FileSystem(key=access_key_id, secret=secret_access_key, requester_pays=True)

    with rio.Env(aws_session):
        # create the landsat scene for that year
        with dask.config.set(scheduler='single-threaded'):
            t0 = time.time()
            print('averaging')
            landsat_ds = scene_seasonal_average(
                path,
                row,
                year,
                access_key_id,
                secret_access_key,
                aws_session,
                core_session,
                fs,
                write_bucket=None,
                bands_of_interest='all',
                landsat_generation='landsat-7',
            )
            t1 = time.time()
            print(f'averaging landsat took {round(t1-t0)} seconds')
            if landsat_ds:
                # reproject from utm to lat/lon
                landsat_zone = landsat_ds.utm_zone_number + landsat_ds.utm_zone_letter
                # sets null value to np.nan
                write_nodata(landsat_ds)
                print('reprojecting')

                data_list, tiles_list, bounding_box_list = reproject_dataset_to_fourthousandth_grid(
                    landsat_ds.astype('float32'), zone=landsat_zone
                )
                del landsat_ds

                dfs = []
                for data, tiles, bounding_box in zip(data_list, tiles_list, bounding_box_list):
                    # add in other datasets
                    data = add_all_variables(data, tiles, year, lat_lon_box=bounding_box).load()
                    df = dataset_to_tabular(data.drop(['spatial_ref']))
                    dfs.append(df)
                    del df
                    del data
                df = pd.concat(dfs)
                df = df.loc[df.ecoregion > 0]
                df['realm'] = df.ecoregion.apply(ECO_TO_REALM_MAP.__getitem__)
            else:
                df = pd.DataFrame({})

        # apply the correct model for each realm
        if len(df) > 0:
            # write input
            if input_write_bucket is not None:
                utils.write_parquet(df, input_write_bucket, access_key_id, secret_access_key)
            rf_result = []
            for realm, sub in df.groupby('realm'):
                rf = m.random_forest_model(
                    realm=realm,
                    df_train=None,
                    df_test=None,
                    output_folder=model_folder,
                    validation_year='none',
                    overwrite=False,
                )
                rf_result.append(make_inference(sub, rf))

            rf_result = pd.concat(rf_result)
            del df
        else:
            rf_result = pd.DataFrame([[np.nan, np.nan, np.nan]], columns=['x', 'y', 'biomass'])

        if output_write_bucket is not None:
            # random forest
            output_filepath = f'{output_write_bucket}/rf/{year}/{path:03d}{row:03d}.parquet'
            utils.write_parquet(rf_result, output_filepath, access_key_id, secret_access_key)
            return ('pass', output_filepath)
        else:
            return rf_result


predict_delayed = dask.delayed(predict)
