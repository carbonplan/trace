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
 
def create_target(min_lat, max_lat, min_lon, max_lon):
    tiles = utils.find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon)
    full_target_ds = utils.open_and_combine_lat_lon_data('gs://carbonplan-climatetrace/intermediates/ecoregions_mask/', 
                                                    tiles=tiles)
    full_target_ds = full_target_ds.rename({'lat': 'y', 'lon': 'x'})
    buffer = 0.01
    target = full_target_ds.sel(y=slice(min_lat-buffer, max_lat+buffer),
                      x=slice(min_lon-buffer, max_lon+buffer))
    return target, tiles

def reproject_dataset(ds, zone=None):
    ds = add_crs_dataset(ds, zone=zone)
    min_lat, max_lat, min_lon, max_lon = check_mins_maxes(ds)
    target, tiles = create_target(min_lat, max_lat, min_lon, max_lon)
    # the numbers aren't too big but if we normalize they might turn into decimals
    reprojected = ds.rio.reproject_match(target).astype('float32') 
    reprojected = reprojected.reflectance.to_dataset(dim='band').drop('spatial_ref').load()
    return reprojected, tiles

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
# @dask.delayed
# def predict():
    # develop inference dataframe
    # load_xgb_model 
    # run model <-- from cindy
    # returns string which is path to dataset it wrote

def load_xgb_model(model_path, local_folder='./'):
    if model_path.startswith('s3'):
        fs = fsspec.get_filesystem_class('s3')()
        model_name = model_path.split('/')[-1]
        fs.get(model_path, local_folder+model_name)
        model_path = local_folder+model_name
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def add_all_variables(data, tiles, year):
    data = load.aster(data, tiles)
    data = load.worldclim(data)
    data = load.igbp(data, tiles, year)
    data = load.treecover2000(tiles, data)
    return data.load()

def make_inference(input_data, model, features):
    """
    input_data is assumed to be a pandas dataframe, and model uses standard sklearn API with .predict
    """
    sub = input_data.dropna(subset=features)
    sub = sub.loc[(~(sub.NDVI == np.inf) & ~(sub.NDII == np.inf))]
    X = sub[features]
    X['burned'] = X.burned.astype(float)
    sub['biomass'] = model.predict(X)
    return sub[['lat', 'lon', 'biomass']]


def predict(model_path, path, row, year, access_key_id, 
                secret_access_key, output_write_bucket=None, 
                                input_write_bucket=None,
                                bands_of_interest='all',
                                season=season):
    model = load_xgb_model(model_path)
    # create the landsat scene for that year
    landsat_ds = scene_seasonal_average(path, row, year, access_key_id, 
                        secret_access_key, write_bucket=None,
                            bands_of_interest='all',
                            season='JJA')
    # add in other datasets
    landsat_zone = landsat_ds.zone_number+landsat_ds.zone_letter
    data, tiles = reproject_dataset(landsat_ds, zone=landsat_zone)
    
    del landsat_ds

    data = add_all_variables(data, tiles, year)
    
    df = dataset_to_tabular(data)

    del data

    if input_write_bucket is not None:
        utils.write_parquet(df, input_write_bucket, access_key_id, secret_access_key)
    
    prediction = make_inference(df, model, features)

    if output_write_bucket is not None:
        utils.write_parquet(prediction, output_write_bucket, access_key_id, secret_access_key)
    
    else:
        return prediction



#     # prediction

#     # save out x/y (specific to row/path scene) and lat/lon (from hansen grid) (plus biomass)

# def repackage():
#     # in_df is list of x/y/biomass
    
# def write_to_zarr():