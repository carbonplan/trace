from .link_biomass_landsat import add_landsat_utm_zone
from .landsat_preprocess import scene_seasonal_average
from ..v1 import utils
import fsspec
import xgboost as xgb 
import xarray as xr
from s3fs import S3FileSystem
import utm
import pandas as pd
from pyproj import CRS
import rioxarray as rio
import pandas as pd
from datetime import datetime, timezone
from carbonplan_trace.v0.data import cat
import dask
from ..v1 import load
import numpy as np
from carbonplan_trace.v1.model import features

fs = S3FileSystem(profile='default', requester_pays=True)



def add_crs_dataset(ds, zone=None):
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
    full_target_ds = utils.open_and_combine_lat_lon_data('gs://carbonplan-climatetrace/intermediates/ecoregions_mask/', 
                                                    tiles=tiles,
                                                    lat_lon_box=[min_lat, max_lat, min_lon, max_lon])
    full_target_ds = full_target_ds.rename({'lat': 'y', 'lon': 'x'})
    buffer = 0.01
    target = full_target_ds.sel(y=slice(min_lat-buffer, max_lat+buffer),
                      x=slice(min_lon-buffer, max_lon+buffer))
    return target, tiles

def reproject_dataset_to_fourthousandth_grid(ds, zone=None):
    ds = add_crs_dataset(ds, zone=zone)
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
    # drop any nan values so we only carry around pixels we have landsat for
    # this will drop both the parts of the dataset that are empty because
    # the landsat scenes might be rotated w.r.t. the x/y grid
    # but will also drop any cloud-masked regions
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
        include variables 'lon', 'lat', and a utm zone called 'landsat_utm_zone'
    
    Returns
    -------
    df : pandas dataframe
        The projected information for each shot
    '''

    return utm.to_latlon(df['x'], df['y'], int(utm_zone_number), utm_zone_letter)


def add_lat_lon_to_table(df, zone_number, zone_letter):
    '''
    Append lat and lon coords, specific to the scene-specific zone.

    Parameters
    ----------
    df : pandas dataframe
        dataframe with columns of bands 
    
    zone : string
        UTM zone retrieved from the landsat scene itself
    
    Returns
    -------
    df : pandas dataframe
        dataframe with columns of bands plus lats/lons

    '''
    lat_lon_info = df.apply(convert_to_lat_lon, args=(zone_number, zone_letter), axis=1).to_list()
    # drop any nan values so we only carry around pixels we have landsat for
    # this will drop both the parts of the dataset that are empty because
    # the landsat scenes might be rotated w.r.t. the x/y grid
    # but will also drop any cloud-masked regions
    projected_column_names = ['lat','lon']
    projection_df = pd.DataFrame(lat_lon_info, columns=projected_column_names, index=df.index)
    updated_df = pd.concat([df, projection_df], axis=1)
    return updated_df

def load_xgb_model(model_path, local_folder='./'):
    if model_path.startswith('s3'):
        fs = fsspec.get_filesystem_class('s3')()
        model_name = model_path.split('/')[-1]
        fs.get(model_path, local_folder+model_name)
        model_path = local_folder+model_name
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def add_all_variables(data, tiles, year, lat_lon_box=None):
    data = load.aster(data, tiles, lat_lon_box=lat_lon_box)
    data = load.worldclim(data)
    data = load.igbp(data, tiles, year, lat_lon_box=lat_lon_box)
    data = load.treecover2000(tiles, data)
    return data.load()

def make_inference(input_data, model, features):
    """
    input_data is assumed to be a pandas dataframe, and model uses standard sklearn API with .predict
    """
    input_data.dropna(subset=features, inplace=True)
    input_data = input_data.loc[(~(input_data.NDVI == np.inf) & ~(input_data.NDII == np.inf))]
    input_data['biomass'] = model.predict(input_data[features])
    return input_data[['x', 'y', 'biomass']]

# @dask.delayed
def predict(model_path, path, row, year, access_key_id, 
                secret_access_key, output_write_bucket=None, 
                                input_write_bucket=None,
                                bands_of_interest='all',
                                season='JJA'):
    model = load_xgb_model(model_path)
    # create the landsat scene for that year
    landsat_ds = scene_seasonal_average(path, row, year, access_key_id, 
                        secret_access_key, write_bucket=None, #'s3://carbonplan-climatetrace/v1/',
                            bands_of_interest='all',
                            season=season)
    ## landsat_ds = xr.open_zarr('s3://carbonplan-climatetrace/v1/45/25/2003/JJA_reflectance.zarr')
    print('landsat loaded')
    # add in other datasets
    landsat_zone = landsat_ds.utm_zone_number+landsat_ds.utm_zone_letter
    data, tiles, bounding_box = reproject_dataset_to_fourthousandth_grid(landsat_ds, zone=landsat_zone)
    print('reprojected!')
    del landsat_ds
    data = add_all_variables(data, tiles, year, lat_lon_box=bounding_box).load()
    print('all variables in!')
    df = dataset_to_tabular(data)
    del data
    print('now in tabular!')
    if input_write_bucket is not None:
        utils.write_parquet(df, input_write_bucket, access_key_id, secret_access_key)
        # df = pd.read_parquet(input_write_bucket)
    prediction = make_inference(df, model, features)
    print('INFERENCE COMPLETE')
    if output_write_bucket is not None:
        utils.write_parquet(prediction, output_write_bucket, access_key_id, secret_access_key)
        print(output_write_bucket)
    else:
        return prediction