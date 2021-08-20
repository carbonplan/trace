from datetime import datetime, datetime as time

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from s3fs import S3FileSystem

from carbonplan_trace.v0.data import cat
from carbonplan_trace.v1.model import features

from ..v1 import utils

# flake8: noqa

fs = S3FileSystem()

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
    'tavg': 100,
    'tmax': 100,
    'tmin': 100,
    'vapr': 100,
    'wind': 100,
}


def aster(ds, tiles, lat_lon_box=None, dtype='int16', chunks_dict=None):
    '''
    Note: ds must have coordinates as x/y and not lon/lat (have different names)
    otherwise the coordinates in the selection
    '''
    full_aster = utils.open_and_combine_lat_lon_data(
        "s3://carbonplan-climatetrace/intermediate/aster/", tiles=tiles, lat_lon_box=lat_lon_box
    )
    if full_aster is not None:
        selected_aster = utils.find_matching_records(
            full_aster, lats=ds.y, lons=ds.x, dtype=dtype
        ).load()
        if chunks_dict is not None:
            selected_aster = selected_aster.chunk(chunks_dict)
        if 'spatial_ref' in selected_aster:
            selected_aster = selected_aster.drop(['spatial_ref'])
        return xr.merge([ds, selected_aster])
    else:
        empty_da = xr.DataArray(np.nan, dims=['x', 'y'], coords=[ds.coords['x'], ds.coords['y']])
        for v in ['elev', 'slope', 'aspect']:
            ds[v] = empty_da
        return ds


def worldclim(ds, dtype='int16', chunks_dict=None):
    mapper = fs.get_mapper(
        's3://carbonplan-climatetrace/v1/data/intermediates/annual_averaged_worldclim.zarr'
    )
    worldclim_ds = xr.open_zarr(mapper, consolidated=True)
    buffer = 0.1
    worldclim_subset = worldclim_ds.sel(
        lon=slice(float(ds.x.min().values) - buffer, float(ds.x.max().values) + buffer),
        lat=slice(float(ds.y.max().values) + buffer, float(ds.y.min().values) - buffer),
    ).load()
    for var in WORLDCLIM_SCALING_FACTORS.keys():
        worldclim_subset[var] = worldclim_subset[var] * WORLDCLIM_SCALING_FACTORS[var]
    worldclim_reprojected = utils.find_matching_records(
        worldclim_subset, ds.y, ds.x, dtype=dtype
    ).load()
    all_vars = worldclim_subset.data_vars
    if chunks_dict is not None:
        worldclim_reprojected = worldclim_reprojected.chunk(chunks_dict)

    for var in all_vars:
        ds[var] = worldclim_reprojected[var]
        del worldclim_reprojected[var]
    return ds


def igbp(data, tiles, year, lat_lon_box=None, dtype='int8'):
    igbp = utils.open_igbp_data(tiles, lat_lon_box=lat_lon_box)
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.y, lons=data.x, years=year, dtype=dtype
    )
    data['igbp'] = igbp_records.igbp.drop(['spatial_ref'])

    del igbp

    return data


def ecoregion(data, tiles, lat_lon_box=None, dtype='int16'):
    eco = utils.open_ecoregion_data(tiles, lat_lon_box=lat_lon_box)
    eco_records = utils.find_matching_records(data=eco, lats=data.y, lons=data.x, dtype=dtype)
    if 'spatial_ref' in eco_records:
        data['ecoregion'] = eco_records.ecoregion.drop(['spatial_ref'])

    del eco

    return data


def treecover2000(data, tiles, lat_lon_box=None, dtype='int8'):
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


def grab_year(df):
    '''
    Access calendar year in timestamp within a dataframe
    '''
    return datetime.fromtimestamp(df['time']).year


def biomass(tiles, year):
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
    no_biomass_data_tiles = ["00N_070E", "20N_120W", "30N_170W", "40N_070W"]
    complete_df = None
    for tile in tiles:
        if tile not in no_biomass_data_tiles:
            file_mapper = fs.get_mapper(
                's3://carbonplan-climatetrace/v1/biomass/{}.zarr'.format(tile)
            )
            ds = xr.open_zarr(file_mapper, consolidated=True)
            df = ds.stack(unique_index=("record_index", "shot_number")).to_dataframe()
            df = df[df['biomass'].notnull()]
            if complete_df is not None:
                complete_df = pd.concat([complete_df, df], axis=0)
            else:
                complete_df = df
    if complete_df is not None:
        complete_df['year'] = complete_df.apply(grab_year, axis=1)
        complete_df = complete_df[complete_df['year'] == year]
    else:
        complete_df = pd.DataFrame({'lat': [], 'lon': [], 'ecoregion': []})
    return complete_df


def training(realm, y0=2003, y1=2010, reload=False, access_key_id=None, secret_access_key=None):
    output_filename = f's3://carbonplan-climatetrace/v1/training/{realm}/all_data.parquet'
    if fs.exists(output_filename) and not reload:
        return pd.read_parquet(output_filename)
    else:
        output = []
        for yr in range(y0, y1):
            folder_name = f's3://carbonplan-climatetrace/v1/training/{realm}/{yr}/'
            files = fs.ls(folder_name)
            for f in files:
                output.append(pd.read_parquet(f's3://{f}'))
        output = pd.concat(output)
        utils.write_parquet(output, output_filename, access_key_id, secret_access_key)
        return output


def tropics(ds, chunks_dict=None):
    """
    tropics is either 1 or 0 (not boolean)
    """
    fp = "s3://carbonplan-climatetrace/inputs/shapes/tropics.shp"
    tropics = gpd.read_file(fp)
    output_da = regionmask.mask_geopandas(
        tropics, numbers='is_tropica', lon_or_obj=ds, lon_name='x', lat_name='y'
    )
    if chunks_dict is not None:
        ds['is_tropics'] = output_da.chunk(chunks_dict)
    else:
        ds['is_tropics'] = output_da

    return ds
