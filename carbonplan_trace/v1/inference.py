from .link_biomass_landsat import add_landsat_utm_zone
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

fs = S3FileSystem(profile='default', requester_pays=True)

def load_biomass(ul_lat, ul_lon):
    '''
    Load in specific biomass tile.
    Parameters
    Parameters
    ----------
    scene_gdf : geopandas geodataframe
        geodataframe whose rows are entries for each row/path scene
    
    Returns
    -------
    scene_gdf : geopandas geodataframe
        same geodataframe but with new column including the USGS-provided,
        scene-specific UTM zone

    '''
    file_mapper = fs.get_mapper('carbonplan-climatetrace/v1/data/intermediates/biomass/{}N_{}W.zarr'.format(ul_lat, ul_lon))

    ds = xr.open_zarr(file_mapper, consolidated=True)
    return ds

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
    reprojected = ds.rio.reproject_match(target)
    reprojected = reprojected.reflectance.to_dataset(dim='band').drop('spatial_ref')
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


# def projection():
#     # same steps as with glas - might already be done (is it in tabular when read in?)

# def append_to_biomass():
#     tack on landsat as extra columns

def load_xgb_model(model_path, local_folder='./'):
    if model_path.startswith('s3'):
        fs = fsspec.get_filesystem_class('s3')()
        model_name = model_path.split('/')[-1]
        fs.get(model_path, local_folder+model_name)
        model_path = local_folder+model_name
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

def nearest_neighbor_variable(data_ds, template_ds):
    nearest_variables = data_ds.sel(lon=template_ds.x, 
                      lat=template_ds.y,
                      method='nearest')
    # print(template_ds.lon)
    # print(template_ds.x)
    # print(nearest_variables.lon)
    # print(nearest_variables.x)

    return nearest_variables

def get_aster(ds, tiles): # tile_names
    '''
    Note: ds must have coordinates as lat/lon and not x/y (have different names)
    otherwise the coordinates will not be turned into 
    '''
    full_aster = utils.open_and_combine_lat_lon_data("gs://carbonplan-climatetrace/intermediates/aster/", 
                                                    tiles=tiles)
    nearest_variables = nearest_neighbor_variable(full_aster, ds)
    
    # print(nearest_variables)#.to_dataframe().drop(['lat', 'lon'], axis=1)
    # aster_records = utils.find_matching_records(data=full_aster, lats=df.y, lons=df.x)
    # print(aster_records)
    # for column in aster_records:
    #     df[column] = aster_records[column]
    return xr.merge([ds, nearest_variables])


def worldclim_seasons(ds):
    # group monthly worldclim data into seasons DJF MAM JJA SON
    days_in_month = {
        1: 31,
        2: 28.25,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }

    months_in_season = [
        (1, [12, 1, 2]),
        (4, [3, 4, 5]),
        (7, [6, 7, 8]),
        (10, [9, 10, 11])
    ]

    month_to_season = {}
    for s, m in months_in_season:
        month_to_season.update({mm: s for mm in m})
        

    monthly_variables = ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']

    seasons = []
    seasonal_data = []
    for season, months in months_in_season: 
        weights = xr.DataArray(
            [days_in_month[m] for m in months],
            dims=['month'],
            coords={'month': months}
        )
        
        seasons.append(season)
        seasonal_data.append(ds[monthly_variables].sel(month=months).weighted(weights).mean(dim='month'))

    seasonal_data = xr.concat(seasonal_data, pd.Index(seasons, name="season"))
    return seasonal_data

def get_worldclim(ds):
    mapper = fsspec.get_mapper('gs://carbonplan-data/raw/worldclim/30s/raster.zarr')
    worldclim = xr.open_zarr(mapper, consolidated=True).rename({"x": "lon", "y": "lat"})
    worldclim_subset = nearest_neighbor_variable(worldclim, ds)
    worldclim_subset = worldclim_subset.drop(['lat', 'lon'])
    seasonal_worldclim = worldclim_seasons(worldclim_subset)
    static_vars = [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
    static_worldclim = worldclim_subset[static_vars]
    return xr.merge([ds, seasonal_worldclim, static_worldclim])


def get_igbp(data, tiles, year):
    igbp = utils.open_igbp_data(tiles)
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.y, lons=data.x, years=year
    )

    data['igbp'] = igbp_records.igbp

    del igbp

    # assert (ds.unique_index == igbp_records.unique_index).mean() == 1
    return data


def get_treecover2000(tiles, data):
    hansen = []
    for tile in tiles:
        lat, lon = utils.get_lat_lon_tags_from_tile_path(tile)
        # get Hansen data
        hansen_tile = cat.hansen_2018(variable='treecover2000', lat=lat, lon=lon).to_dask()
        hansen_tile = hansen_tile.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
        hansen.append(hansen_tile.to_dataset(name='treecover2000', promote_attrs=True))
    hansen = xr.combine_by_coords(hansen, combine_attrs="drop_conflicts").chunk(
        {'lat': 2000, 'lon': 2000}
    )

    hansen_records = utils.find_matching_records(data=hansen, lats=data.y, lons=data.x)
    # assert (data.unique_index == hansen_records.unique_index).mean() == 1
    data['treecover2000_mean'] = hansen_records['treecover2000']

    del hansen

    return data
    
# def predict(df ):
#     # load model
    
#     # prediction
#     # save out x/y (specific to row/path scene) and lat/lon (from hansen grid) (plus biomass)

# def repackage():
#     # in_df is list of x/y/biomass
    
# def write_to_zarr():