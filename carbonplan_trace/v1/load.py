import xarray as xr
from ..v1 import utils
import fsspec
from carbonplan_trace.v0.data import cat

def aster(ds, tiles, lat_lon_box=None, dtype='int16'):
    '''
    Note: ds must have coordinates as lat/lon and not x/y (have different names)
    otherwise the coordinates will not be turned into 
    '''
    full_aster = utils.open_and_combine_lat_lon_data("gs://carbonplan-climatetrace/intermediates/aster/", 
                                                    tiles=tiles, 
                                                    lat_lon_box=lat_lon_box)
    selected_aster = utils.find_matching_records(full_aster, lats=ds.y, lons=ds.x, dtype=dtype).load().drop(['spatial_ref'])
    return xr.merge([ds, selected_aster])

def worldclim(ds, dtype='int16'):
    mapper = fsspec.get_mapper('s3://carbonplan-climatetrace/v1/data/intermediates/annual_averaged_worldclim.zarr')
    worldclim_ds = xr.open_zarr(mapper, consolidated=True)
    worldclim_subset = worldclim_ds.sel(lon=slice(float(ds.x.min().values), float(ds.x.max().values)),
                            lat=slice(float(ds.y.max().values), float(ds.y.min().values))
                                    ).load()
    worldclim_reprojected = utils.find_matching_records(worldclim_subset, ds.y, ds.x, dtype=dtype).load()
    for var in worldclim_reprojected.data_vars:
        ds[var] = worldclim_reprojected[var]
        del worldclim_reprojected[var]
    return ds


def igbp(data, tiles, year, lat_lon_box=None, dtype='int8'):
    igbp = utils.open_igbp_data(tiles, lat_lon_box=lat_lon_box)
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.y, lons=data.x, years=year, dtype=dtype
    )
    data['burned'] = igbp_records.igbp.drop(['spatial_ref'])

    del igbp

    return data


def treecover2000(tiles, data, lat_lon_box=None, dtype='int8'):
    hansen = []
    for tile in tiles:
        lat, lon = utils.get_lat_lon_tags_from_tile_path(tile)
        # get Hansen data
        hansen_tile = cat.hansen_change(variable='treecover2000', lat=lat, lon=lon).to_dask()
        hansen_tile = hansen_tile.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
        if lat_lon_box is not None:
            [min_lat, max_lat, min_lon, max_lon] = lat_lon_box
            hansen_tile = hansen_tile.sel(lat=slice(min_lat, max_lat),
                        lon=slice(min_lon, max_lon))
        hansen.append(hansen_tile.to_dataset(name='treecover2000', promote_attrs=True))

    hansen = xr.combine_by_coords(hansen, combine_attrs="drop_conflicts").chunk(
        {'lat': 2000, 'lon': 2000}
    )

    hansen_records = utils.find_matching_records(data=hansen, lats=data.y, lons=data.x, dtype=dtype)
    data['treecover2000_mean'] = hansen_records['treecover2000']

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
    # TODO need to fix this to open in the same way as for the other datasets (by passing the bounding box)
    file_mapper = fs.get_mapper('carbonplan-climatetrace/v1/data/intermediates/biomass/{}N_{}W.zarr'.format(ul_lat, ul_lon))

    ds = xr.open_zarr(file_mapper, consolidated=True)
    return ds