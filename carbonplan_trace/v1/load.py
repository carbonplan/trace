import xarray as xr
from ..v1 import utils
import fsspec


def nearest_neighbor_variable(data_ds, template_ds):
    nearest_variables = data_ds.sel(lon=template_ds.x, 
                      lat=template_ds.y,
                      method='nearest')

    return nearest_variables.astype('int16')

def aster(ds, tiles): # tile_names
    '''
    Note: ds must have coordinates as lat/lon and not x/y (have different names)
    otherwise the coordinates will not be turned into 
    '''
    full_aster = utils.open_and_combine_lat_lon_data("gs://carbonplan-climatetrace/intermediates/aster/", 
                                                    tiles=tiles)
    nearest_variables = nearest_neighbor_variable(full_aster, ds).load().drop(['spatial_ref', 'lat', 'lon'])
    return xr.merge([ds, nearest_variables])

def worldclim(ds, timestep='annual'):
    if timestep=='monthly':
        mapper = fsspec.get_mapper('gs://carbonplan-data/raw/worldclim/30s/raster.zarr')
        worldclim_ds = xr.open_zarr(mapper, consolidated=True).rename({"x": "lon", "y": "lat"})

    elif timestep=='annual':
        mapper = fsspec.get_mapper('s3://carbonplan-climatetrace/v1/data/intermediates/annual_averaged_worldclim.zarr')
        worldclim_ds = xr.open_zarr(mapper, consolidated=True)
    worldclim_subset = worldclim_ds.sel(lon=slice(float(ds.x.min().values), float(ds.x.max().values)),
                            lat=slice(float(ds.y.max().values), float(ds.y.min().values))
                                    ).load()
    worldclim_reprojected = nearest_neighbor_variable(worldclim_subset, ds).load()
    worldclim_reprojected = worldclim_reprojected.drop(['lat', 'lon'])
    if timestep=='monthly':
        seasonal_worldclim = worldclim_seasons(worldclim_reprojected).load()
        print('seasonal!')
        static_vars = [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
        static_worldclim = worldclim_reprojected[static_vars].astype('int8')
        return xr.merge([ds, seasonal_worldclim, static_worldclim])
    elif timestep=='annual':
        return xr.merge([ds, worldclim_reprojected])


def igbp(data, tiles, year):
    igbp = utils.open_igbp_data(tiles)
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.y, lons=data.x, years=year
    )
    data['igbp'] = igbp_records.igbp.drop(['spatial_ref']).astype('int8')

    del igbp

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
    data['treecover2000_mean'] = hansen_records['treecover2000'].astype('int8')

    del hansen

    return data

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