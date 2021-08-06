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

