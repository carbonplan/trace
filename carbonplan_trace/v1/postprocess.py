import numpy as np
import xarray as xr
import fsspec
import pandas as pd

from ..v1 import load, utils


def compile_df_for_tile(ul_lat, ul_lon, year, tile_degree_size=2):
    fs = fsspec.get_filesystem_class('s3')()

    scene_ids = utils.grab_all_scenes_in_tile(ul_lat,ul_lon, 
                    tile_degree_size=tile_degree_size)
    list_of_parquet_paths = [f's3://carbonplan-climatetrace/v1/inference/rf/{year}/{path:03d}{row:03d}.parquet'  for [path, row] in scene_ids]
    dfs = [pd.read_parquet(f's3://{path}') for path in list_of_parquet_paths]
    compiled_df = pd.concat(dfs)
    return compiled_df

def turn_point_cloud_to_grid(df, tile_degree_size):
    df.x = df.x.round(6)
    df.y = df.y.round(6)

    lats = np.arange(df.y.min(), df.y.max(), .00025).round(6)
    lons = np.arange(df.x.min(), df.x.max(), .00025).round(6)
    df = df.groupby(['x', 'y']).mean().reset_index()
    pivot = df.pivot(
        columns="x", index="y", values="biomass"
    )
    reindexed = pivot.reindex(index=lats, columns=lons)
    ds_grid = xr.DataArray(
        data=reindexed.values,
        dims=["y", "x"],
        coords=[lats, lons],
    )
    ds_grid = ds_grid.to_dataset(name="biomass", promote_attrs=True)
    ds_grid = ds_grid.rename({'x': 'lon', 'y': 'lat'})
    return ds_grid

def trim_ds(ul_lat, ul_lon, tile_degree_size, ds):
    ds = ds.sel(x=slice(ul_lon, ul_lon+2), y=slice(ul_lat - 2, ul_lat))
    return ds

def merge_all_scenes_in_tile(ul_lat, ul_lon, year, tile_degree_size=2):
    df = compile_df_for_tile(ul_lat, ul_lon, year)
    ds = turn_point_cloud_to_grid(df, tile_degree_size)
    ds = trim_ds(ul_lat, ul_lon, tile_degree_size, ds)
    return ds

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
    ds = ds.interpolate_na(dim='time', method='linear', fill_value="extrapolate")
    ds = ds.interpolate_na(dim='lon', method='linear', fill_value="extrapolate")
    ds = ds.interpolate_na(dim='lat', method='linear', fill_value="extrapolate")
    
    ds['biomass'] = ds.biomass.clip(min=min_biomass, max=max_biomass)
    
    return ds 

def apply_forest_mask(biomass_ds, lat_lon_box=None):
    """
    Set biomass values in non-forests areas to null based on MODIS MCD12Q1 data with IGBP legend 
    Forest is defined as within class 1-5, 8, and 9 
    (classes with tree cover greater than 10% and canopy greater than 2m in height)
    """
    igbp = open_global_igbp_data(lat_lon_box=lat_lon_box)
    forest_mask = igbp.igbp.isin([1, 2, 3, 4, 5, 8, 9]).any(dim='year')
    biomass_ds['forest_mask'] = utils.find_matching_records(
        data=forest_mask, lats=biomass_ds.lat, lons=biomass_ds.lon
    )
    biomass_ds = biomass_ds.where(biomass_ds.forest_mask)
    
    return biomass_ds 

def calculate_belowground_biomass(ds):
    """
    1. rename biomass to agb 
    2. calculate bgb from agb based on power law relationship published by Mokany et al (2006) Eq 1
    "Critical analysis of root: Shoot ratios in terrestrial biomes"
    """
    ds = ds.rename({'biomass': 'agb'})
    ds['bgb'] = 0.489 * (ds.agb ** 0.890)
    
    return ds 

def calculate_dead_wood_and_litter(ds):
    """
    1. load FAO ecozone (converted into tropics/not tropics shapefile), elevation, and precipitation
    2. for each pixel, figure out the dead wood and litter fraction based on "
    Methodological Tool: Estimation of Carbon Stocks and Change in Carbon Stocks in Dead Wood and Litter in A/R CDM Project Activities (UNFCCC, 2013);
    https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ ar-am-tool-12-v3.0.pdf" 
    3. calculate dead wood and litter 
    """
    ds = load.tropics(ds)
    ds = load.aster(ds, tiles, lat_lon_box=lat_lon_box)
    ds = load.worldclim(ds)
    
    dead_wood = xr.DataArray(0, dims=['lat', 'lon', 'year'], coords=[ds.coords['lat'], ds.coords['lon'], ds.coords['year']])
    litter = xr.DataArray(0, dims=['lat', 'lon', 'year'], coords=[ds.coords['lat'], ds.coords['lon'], ds.coords['year']])
    
    # tropic, elevation < 2000m, precip < 1000mm
    dead_wood = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 < 1000), x=(ds.agb * 0.02), y=dead_wood)
    litter = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 < 1000), x=(ds.agb * 0.04), y=litter)

    # tropic, elevation < 2000m, precip 1000-1600mm
    dead_wood = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1000) & (ds.BIO12 < 1600), x=(ds.agb * 0.01), y=dead_wood)
    litter = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1000) & (ds.BIO12 < 1600), x=(ds.agb * 0.01), y=litter)
    
    # tropic, elevation < 2000m, precip > 1600mm
    dead_wood = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1600), x=(ds.agb * 0.06), y=dead_wood)
    litter = xr.where((ds.is_tropics == 1) & (ds.elev < 2000) & (ds.BIO12 >= 1600), x=(ds.agb * 0.01), y=litter)
    
    # tropic, elevation >= 2000m
    dead_wood = xr.where((ds.is_tropics == 1) & (ds.elev >= 2000), x=(ds.agb * 0.07), y=dead_wood)
    litter = xr.where((ds.is_tropics == 1) & (ds.elev >= 2000), x=(ds.agb * 0.01), y=litter)
    
    # non tropic 
    dead_wood = xr.where((ds.is_tropics == 0), x=(ds.agb * 0.08), y=dead_wood)
    litter = xr.where((ds.is_tropics == 0), x=(ds.agb * 0.04), y=litter)
    
    ds['dead_wood'] = dead_wood
    ds['litter'] = litter
    
    return ds 


def fillna_mask_and_calc_carbon_pools(data):
    """
    input = 3D merged result with lat, lon, year and biomass being the only data variable
    output = input data with more data variables for other carbon pools, with nulls filled and masked with forest land cover 
    """
    min_lat = data.lat.min().values
    max_lat = data.lat.max().values
    min_lon = data.lon.min().values
    max_lon = data.lon.max().values
    lat_lon_box = min_lat, max_lat, min_lon, max_lon
    
    data = fill_nulls(data.load())
    data = apply_forest_mask(data, lat_lon_box=lat_lon_box)
    data = calculate_belowground_biomass(data)
    data = calculate_deadwood_and_litter(data)

    return data
