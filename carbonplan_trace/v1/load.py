import fsspec
import xarray as xr
from s3fs import S3FileSystem

from carbonplan_trace.v0.data import cat

from ..v1 import utils

# flake8: noqa

fs = S3FileSystem(profile='default', requester_pays=True)
WORLDCLIM_SCALING_FACTORS = {
    'BIO01': 100,
    'BIO02': 100,
    'BIO03': 1,
    'BIO04': 1,
    'BIO05': 100,
    'BIO06': 100,
    'BIO07': 100,
    'BIO08': 100,
    'BIO09': 100,
    'BIO10': 100,
    'BIO11': 100,
    'BIO12': 1,
    'BIO13': 1,
    'BIO14': 1,
    'BIO15': 100,
    'BIO16': 1,
    'BIO17': 1,
    'BIO18': 1,
    'BIO19': 1,
}


def aster(ds, tiles, lat_lon_box=None, dtype='int16'):
    '''
    Note: ds must have coordinates as lat/lon and not x/y (have different names)
    otherwise the coordinates will not be turned into
    '''
    print('load aster')
    full_aster = utils.open_and_combine_lat_lon_data(
        "gs://carbonplan-climatetrace/intermediates/aster/", tiles=tiles, lat_lon_box=lat_lon_box
    )
    print('add reprojected aster')
    selected_aster = (
        utils.find_matching_records(full_aster, lats=ds.y, lons=ds.x, dtype=dtype)
        .load()
        .drop(['spatial_ref'])
    )
    print('merging aster')
    return xr.merge([ds, selected_aster])


def worldclim(ds, dtype='int16'):
    mapper = fsspec.get_mapper(
        's3://carbonplan-climatetrace/v1/data/intermediates/annual_averaged_worldclim.zarr'
    )
    print('loading worldlcim')
    worldclim_ds = xr.open_zarr(mapper, consolidated=True).astype(dtype)
    print('subsetting')
    worldclim_subset = worldclim_ds.sel(
        lon=slice(float(ds.x.min().values), float(ds.x.max().values)),
        lat=slice(float(ds.y.max().values), float(ds.y.min().values)),
    ).load()
    print('scaling')
    for var in WORLDCLIM_SCALING_FACTORS.keys():
        worldclim_subset[var] = worldclim_subset[var] * WORLDCLIM_SCALING_FACTORS[var]
    print('reproejcting')
    worldclim_reprojected = utils.find_matching_records(
        worldclim_subset, ds.y, ds.x, dtype=dtype
    ).load()
    print('adding to dataset and deleting')
    all_vars = worldclim_subset.data_vars

    for var in all_vars:
        ds[var] = worldclim_reprojected[var]
        del worldclim_reprojected[var]
    return ds


def igbp(data, tiles, year, lat_lon_box=None, dtype='int8'):
    print('loading igbp')
    igbp = utils.open_igbp_data(tiles, lat_lon_box=lat_lon_box)
    print('reproject igbp')
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.y, lons=data.x, years=year, dtype=dtype
    )
    print('add igbp')
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
            hansen_tile = hansen_tile.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
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
    file_mapper = fs.get_mapper(
        'carbonplan-climatetrace/v1/data/intermediates/biomass/{}N_{}W.zarr'.format(ul_lat, ul_lon)
    )

    ds = xr.open_zarr(file_mapper, consolidated=True)
    return ds
