import gc
import os

import boto3
import dask
import fsspec
import numpy as np
import rasterio as rio
import rioxarray
import utm
import xarray as xr
import xgboost as xgb
from pyproj import CRS
from rasterio.session import AWSSession
from s3fs import S3FileSystem

from carbonplan_trace.v1.landsat_preprocess import scene_seasonal_average
from carbonplan_trace.v1.model import features

from ..v1 import load, utils

# flake8: noqa

fs = S3FileSystem(requester_pays=True)


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
    full_target_ds = utils.open_and_combine_lat_lon_data(
        'gs://carbonplan-climatetrace/intermediates/ecoregions_mask/',
        tiles=tiles,
        lat_lon_box=[min_lat, max_lat, min_lon, max_lon],
    )
    full_target_ds = full_target_ds.rename({'lat': 'y', 'lon': 'x'})
    buffer = 0.01
    target = full_target_ds.sel(
        y=slice(min_lat - buffer, max_lat + buffer), x=slice(min_lon - buffer, max_lon + buffer)
    )
    return target, tiles


def reproject_dataset_to_fourthousandth_grid(ds, zone=None):
    ds = write_crs_dataset(ds, zone=zone)
    min_lat, max_lat, min_lon, max_lon = check_mins_maxes(ds)
    target, tiles = create_target_grid(min_lat, max_lat, min_lon, max_lon)
    # the numbers aren't too big but if we normalize they might turn into decimals
    reprojected = ds.rio.reproject_match(target).load()
    del ds
    return reprojected, tiles, [min_lat, max_lat, min_lon, max_lon]


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
    df = df.dropna().reset_index()
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


def load_xgb_model(model_path, fs, local_folder='./'):
    cwd = os.getcwd()
    if model_path.startswith('s3'):
        model_name = model_path.split('/')[-1]
        new_model_path = ('/').join([cwd, model_name])
        fs.get(model_path, new_model_path)
        model_path = new_model_path

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    return model


def add_all_variables(data, tiles, year, lat_lon_box=None):
    data = load.aster(data, tiles, lat_lon_box=lat_lon_box)
    data = load.worldclim(data)
    data = load.igbp(data, tiles, year, lat_lon_box=lat_lon_box)
    data = load.treecover2000(tiles, data)
    return data


def make_inference(input_data, model, features):
    """
    input_data is assumed to be a pandas dataframe, and model uses standard sklearn API with .predict
    """
    input_data = input_data.dropna(subset=features)
    input_data = input_data.loc[(~(input_data.NDVI == np.inf) & ~(input_data.NDII == np.inf))]
    gc.collect()
    input_data['biomass'] = model.predict(input_data[features])
    return input_data[['x', 'y', 'biomass']]


def predict(
    model_path,
    path,
    row,
    year,
    access_key_id,
    secret_access_key,
    output_write_bucket=None,
    input_write_bucket=None,
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

            model = load_xgb_model(model_path, fs)
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
                write_bucket=None,  #'s3://carbonplan-climatetrace/v1/',
                bands_of_interest='all',
                season=season,
            )
            # add in other datasets
            landsat_zone = landsat_ds.utm_zone_number + landsat_ds.utm_zone_letter
            data, tiles, bounding_box = reproject_dataset_to_fourthousandth_grid(
                landsat_ds, zone=landsat_zone
            )
            del landsat_ds
            data = add_all_variables(data, tiles, year, lat_lon_box=bounding_box).load()
            df = dataset_to_tabular(data)
            del data
            if input_write_bucket is not None:
                utils.write_parquet(df, input_write_bucket, access_key_id, secret_access_key)
            prediction = make_inference(df, model, features)
            del df
            if output_write_bucket is not None:
                utils.write_parquet(
                    prediction, output_write_bucket, access_key_id, secret_access_key
                )
                print(output_write_bucket)
            else:
                return prediction


predict_delayed = dask.delayed(predict)
