import boto3
import dask
import fsspec
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
from prefect import task
from rasterio.session import AWSSession

from carbonplan_trace.v0.data import cat

from ..v1 import load, utils


def compile_df_for_tile(ul_lat, ul_lon, year, tile_degree_size=2):
    scene_ids = utils.grab_all_scenes_in_tile(ul_lat, ul_lon, tile_degree_size=tile_degree_size)
    list_of_parquet_paths = [
        f's3://carbonplan-climatetrace/v1/inference/rf/{year}/{path:03d}{row:03d}.parquet'
        for [path, row] in scene_ids
    ]

    dfs = []
    for path in list_of_parquet_paths:
        temp = pd.read_parquet(f's3://{path}').round(6)
        temp = temp.loc[
            (temp.y >= ul_lat - tile_degree_size)
            & (temp.y <= ul_lat)
            & (temp.x >= ul_lon)
            & (temp.x <= ul_lon + tile_degree_size)
        ]
        temp['biomass'] = temp.biomass.astype('float32')
        dfs.append(temp)
        del temp

    compiled_df = pd.concat(dfs)
    del dfs
    compiled_df['biomass'] = compiled_df['biomass'].astype('float32')  # convert to float32

    return compiled_df


def turn_point_cloud_to_grid(df, ul_lat, ul_lon, tile_degree_size):
    df.x = df.x.round(6)
    df.y = df.y.round(6)
    pixel_size = 0.00025
    lats = np.arange(
        ul_lat - tile_degree_size + pixel_size / 2, ul_lat - pixel_size / 2, pixel_size
    ).round(6)
    lons = np.arange(
        ul_lon + pixel_size / 2, ul_lon + tile_degree_size - pixel_size / 2, pixel_size
    ).round(6)
    df = df.groupby(['x', 'y']).mean().reset_index()

    pivot = df.pivot(columns="x", index="y", values="biomass")
    del df
    reindexed = pivot.reindex(index=lats, columns=lons)

    ds_grid = xr.DataArray(
        data=reindexed.values,
        dims=["y", "x"],
        coords=[lats, lons],
    ).astype('float32')
    del reindexed
    ds_grid = ds_grid.to_dataset(name="biomass", promote_attrs=True)
    return ds_grid


def trim_ds(ul_lat, ul_lon, tile_degree_size, ds):
    ds = ds.sel(
        x=slice(ul_lon, ul_lon + tile_degree_size), y=slice(ul_lat - tile_degree_size, ul_lat)
    )
    return ds


def merge_all_scenes_in_tile(ul_lat, ul_lon, year, tile_degree_size=2):
    df = compile_df_for_tile(ul_lat, ul_lon, year)
    ds = turn_point_cloud_to_grid(df, ul_lat, ul_lon, tile_degree_size)
    del df
    ds = trim_ds(ul_lat, ul_lon, tile_degree_size, ds)
    return ds


def biomass_tile_timeseries(ul_lat, ul_lon, year0, year1, tile_degree_size=2):
    ds_list = []
    for year in np.arange(year0, year1):
        ds_list.append(
            merge_all_scenes_in_tile(ul_lat, ul_lon, year, tile_degree_size=tile_degree_size)
        )
    biomass_timeseries = xr.concat(ds_list, dim='time')

    biomass_timeseries = biomass_timeseries.assign_coords(
        # {'time': pd.date_range(str(year0), str(year1), freq='A')}
        {'time': np.arange(year0, year1)}
    )
    return biomass_timeseries


def initialize_empty_dataset(ul_lat_tag, ul_lon_tag, year0, year1, write_tile_metadata=True):

    sample_hansen_tile = (
        cat.hansen_change(variable='treecover2000', lat=ul_lat_tag, lon=ul_lon_tag)
        .to_dask()
        .drop_vars('band')
        .squeeze()
    )
    sample_hansen_tile = sample_hansen_tile.rename({'x': 'lon', 'y': 'lat'})
    # sample tiles have lats in descending order- we'll reorder them to make the
    # index-location-based region to_zarr writing at the end of the post-processing
    # more straightforward
    sample_hansen_tile = sample_hansen_tile.reindex(lat=sample_hansen_tile.lat[::-1])
    ds_list = []
    for year in np.arange(year0, year1):
        ds_list.append(sample_hansen_tile)
    timeseries = xr.concat(ds_list, dim='time')
    timeseries = timeseries.assign_coords({'time': pd.date_range(str(year0), str(year1), freq='A')})
    ds = timeseries.to_dataset(name='AGB')
    for variable in ['BGB', 'dead_wood', 'litter']:
        ds[variable] = ds['AGB']
    path = 's3://carbonplan-climatetrace/v1/results/tiles/{}_{}.zarr'.format(ul_lat_tag, ul_lon_tag)
    mapper = fsspec.get_mapper(path)
    if write_tile_metadata:
        ds.astype('float32').to_zarr(mapper, mode='w', compute=False)
    return path


def fill_nulls(ds):
    """
    fill gaps within the biomass dataset by xarray interpolate_na with linear method
    we first fill gaps in time, then lon, then lat. this is because we expect biomass to be relatively
    more stable temporally versus spatially, and because larger gaps are more likely to exist along lat
    dimension (due to landsat orbiting pattern), thus filling first in lon then in lat will likely result
    in shorter gap fills.
    """
    min_biomass = ds.biomass.min().values
    max_biomass = ds.biomass.max().values
    with dask.config.set(scheduler="threads"):
        ds = ds.interpolate_na(dim='time', method='linear')  # , bounds_error=False)
    ds = ds.interpolate_na(dim='x', method='linear', fill_value="extrapolate")

    ds = ds.interpolate_na(dim='y', method='linear', fill_value="extrapolate")

    ds['biomass'] = ds.biomass.clip(min=min_biomass, max=max_biomass)

    return ds


def apply_forest_mask(biomass_ds, lat_lon_box=None, chunks_dict=None):
    """
    Set biomass values in non-forests areas to null based on MODIS MCD12Q1 data with IGBP legend
    Forest is defined as within class 1-5, 8, and 9
    (classes with tree cover greater than 10% and canopy greater than 2m in height)
    """
    igbp = utils.open_global_igbp_data(lat_lon_box=lat_lon_box)
    forest_mask = igbp.igbp.isin([1, 2, 3, 4, 5, 8, 9]).any(dim='year')
    biomass_ds['forest_mask'] = utils.find_matching_records(
        data=forest_mask, lats=biomass_ds.y, lons=biomass_ds.x
    )  # .chunk(chunks_dict)
    biomass_ds = biomass_ds.where(biomass_ds.forest_mask)

    return biomass_ds.astype('float32')


def calculate_belowground_biomass(ds):
    """
    1. rename biomass to agb
    2. calculate bgb from agb based on power law relationship published by Mokany et al (2006) Eq 1
    "Critical analysis of root: Shoot ratios in terrestrial biomes"
    """
    ds = ds.rename({'biomass': 'AGB'})
    ds['BGB'] = 0.489 * (ds.AGB ** 0.890)

    return ds.astype('float32')


def calculate_dead_wood_and_litter(ds, tiles, chunks_dict, lat_lon_box=None):
    """
    1. load FAO ecozone (converted into tropics/not tropics shapefile), elevation, and precipitation
    2. for each pixel, figure out the dead wood and litter fraction based on "
    Methodological Tool: Estimation of Carbon Stocks and Change in Carbon Stocks in Dead Wood and Litter in A/R CDM Project Activities (UNFCCC, 2013);
    https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ ar-am-tool-12-v3.0.pdf"
    3. calculate dead wood and litter
    """
    ds = load.tropics(ds, chunks_dict=chunks_dict)
    ds = load.aster(ds, tiles, chunks_dict=chunks_dict, lat_lon_box=lat_lon_box)
    ds = load.worldclim(ds, chunks_dict=chunks_dict)
    # ds = ds.chunk(chunks_dict)
    dead_wood = (
        xr.DataArray(
            0,
            dims=['y', 'x', 'time'],
            coords=[ds.coords['y'], ds.coords['x'], ds.coords['time']],
        )
        .astype('float32')
        .chunk(chunks_dict)
    )
    litter = (
        xr.DataArray(
            0,
            dims=['y', 'x', 'time'],
            coords=[ds.coords['y'], ds.coords['x'], ds.coords['time']],
        )
        .astype('float32')
        .chunk(chunks_dict)
    )

    # tropic, elevation < 2000m, precip < 1000mm
    dead_wood = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 < 1000), x=(ds.AGB * 0.02), y=dead_wood
    )
    litter = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 < 1000), x=(ds.AGB * 0.04), y=litter
    )

    # tropic, elevation < 2000m, precip 1000-1600mm
    dead_wood = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1000) & (ds.BIO12 < 1600),
        x=(ds.AGB * 0.01),
        y=dead_wood,
    )
    litter = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1000) & (ds.BIO12 < 1600),
        x=(ds.AGB * 0.01),
        y=litter,
    )

    # tropic, elevation < 2000m, precip > 1600mm
    dead_wood = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1600), x=(ds.AGB * 0.06), y=dead_wood
    )
    litter = xr.where(
        (ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1600), x=(ds.AGB * 0.01), y=litter
    )

    # tropic, elevation >= 2000m
    dead_wood = xr.where((ds.is_tropics == 1) & (ds.elev >= 2000), x=(ds.AGB * 0.07), y=dead_wood)
    litter = xr.where((ds.is_tropics == 1) & (ds.elev >= 2000), x=(ds.AGB * 0.01), y=litter)

    # non tropic
    dead_wood = xr.where((ds.is_tropics == 0), x=(ds.AGB * 0.08), y=dead_wood)
    litter = xr.where((ds.is_tropics == 0), x=(ds.AGB * 0.04), y=litter)

    ds['dead_wood'] = dead_wood.astype('float32')
    ds['litter'] = litter.astype('float32')
    ds = ds[['AGB', 'BGB', 'dead_wood', 'litter']]
    return ds


def fillna_mask_and_calc_carbon_pools(data, chunks_dict):
    """
    input = 3D merged result with lat, lon, year and biomass being the only data variable
    output = input data with more data variables for other carbon pools, with nulls filled and masked with forest land cover
    """
    min_lat = data.y.min().values
    max_lat = data.y.max().values
    min_lon = data.x.min().values
    max_lon = data.x.max().values
    lat_lon_box = min_lat, max_lat, min_lon, max_lon
    # get lat lon tags
    tiles = utils.find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon)

    data = fill_nulls(data.load()).chunk(chunks_dict)
    data = apply_forest_mask(data, lat_lon_box=lat_lon_box, chunks_dict=chunks_dict)
    data = calculate_belowground_biomass(data)
    data = calculate_dead_wood_and_litter(
        data, tiles, chunks_dict=chunks_dict, lat_lon_box=lat_lon_box
    )
    data = data.rename({'x': 'lon', 'y': 'lat'})
    data = data.transpose('time', 'lat', 'lon')
    return data


def write_to_log(string, log_path, access_key_id, secret_access_key):
    fs = fsspec.get_filesystem_class('s3')(key=access_key_id, secret=secret_access_key)
    with fs.open(log_path, 'w') as f:
        f.write(string)


def postprocess_subtile(parameters_dict):
    min_lat = parameters_dict['MIN_LAT']
    min_lon = parameters_dict['MIN_LON']
    lat_increment = parameters_dict['LAT_INCREMENT']
    lon_increment = parameters_dict['LON_INCREMENT']
    year0 = parameters_dict['YEAR_0']
    year1 = parameters_dict['YEAR_1']
    tile_degree_size = parameters_dict['TILE_DEGREE_SIZE']
    data_path = parameters_dict['DATA_PATH']
    access_key_id = parameters_dict['ACCESS_KEY_ID']
    secret_access_key = parameters_dict['SECRET_ACCESS_KEY']
    chunks_dict = parameters_dict['CHUNKS_DICT']

    subtile_ul_lat = min_lat + lat_increment + tile_degree_size
    subtile_ul_lon = min_lon + lon_increment
    core_session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name='us-west-2',
    )

    aws_session = AWSSession(core_session, requester_pays=True)
    log_path = f's3://carbonplan-climatetrace/v1/postprocess_log/{min_lat}_{min_lon}_{lat_increment}_{lon_increment}.txt'
    # we initialize the fs here to ensure that the worker has the correct permissions
    # in order to write
    fs = fsspec.get_filesystem_class('s3')(key=access_key_id, secret=secret_access_key)
    data_mapper = fs.get_mapper(data_path)

    ds = biomass_tile_timeseries(
        subtile_ul_lat, subtile_ul_lon, year0, year1, tile_degree_size=tile_degree_size
    )

    with rio.Env(aws_session):

        # add all other postprocessing
        ds = fillna_mask_and_calc_carbon_pools(ds, chunks_dict=chunks_dict)
        # add the timestamps back in to conform with template
        ds = ds.assign_coords({'time': pd.date_range(str(year0), str(year1), freq='A')})
        # rechunk aligning with the template file
        ds = ds.chunk({'lat': 4000, 'lon': 4000, 'time': 1})
        for v in ds.data_vars:
            if 'chunks' in ds[v].encoding:
                del ds[v].encoding['chunks']
        ds.to_zarr(
            data_mapper,
            mode='a',
            region={
                "lat": slice(lat_increment * 4000, (lat_increment + tile_degree_size) * 4000),
                'lon': slice(lon_increment * 4000, (lon_increment + tile_degree_size) * 4000),
                'time': slice(0, year1 - year0),
            },
        )

    write_to_log('done', log_path, access_key_id, secret_access_key)


postprocess_delayed = dask.delayed(postprocess_subtile)
postprocess_task = task(postprocess_subtile, tags=["dask-resource:workertoken=1"])
