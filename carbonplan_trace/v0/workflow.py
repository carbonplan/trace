#!/usr/bin/env python3

import dask
import fsspec
import geopandas
import numcodecs
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from dask.distributed import Client

from carbonplan_trace.metadata import get_cf_global_attrs
from carbonplan_trace.tiles import tiles
from carbonplan_trace.utils import zarr_is_complete
from carbonplan_trace.v0.core import calc_emissions, coarsen_emissions
from carbonplan_trace.v0.data.load import open_hansen_change_tile

tile_template = "s3://carbonplan-climatetrajce/v0.1/tiles/30m/{tile_id}.zarr"
coarse_tile_template = "s3://carbonplan-climatetrace/v0.1/tiles/3000m/{tile_id}.zarr"
coarse_full_template = "s3://carbonplan-climatetrace/v0.1/global/3000m/raster.zarr"
shapes_file = 's3://carbonplan-climatetrace/country-shapes.json'
# TODO: update to remote location
shapes_file = '../../notebooks/countries.shp'
rollup_template = 's3://carbonplan-climatetrace/v0.1/country_rollups.csv'
chunks = {"lat": 4000, "lon": 4000, "year": 2}
coarse_chunks = {"lat": 400, "lon": 400, "year": -1}
COARSENING_FACTOR = 100


def process_one_tile(tile_id):
    """
    Given lat and lon to select a region, calculate the
    corresponding emissions for each year from 2001 to 2020

    Parameters
    ----------
    tile_id : str
        String specifying which 10x10-degree tile to process, e.g. `50N_120W`

    Returns
    -------
    url : string
        Url where a processed tile is located
    """
    print(tile_id)
    return_status = 'skipped'
    encoding = {"emissions": {"compressor": numcodecs.Blosc()}}

    # calc emissions
    url = tile_template.format(tile_id=tile_id)
    mapper = fsspec.get_mapper(url)
    if not zarr_is_complete(mapper):

        lat, lon = tile_id.split('_')
        ds = open_hansen_change_tile(lat, lon)

        out = calc_emissions(ds).to_dataset(name='emissions')  # .chunk(chunks)
        out.attrs.update(get_cf_global_attrs())

        print(out)
        print(out.nbytes / 1e9)
        # TODO: add metadata to emissions variable
        out.to_zarr(mapper, encoding=encoding, mode="w", consolidated=True)
        return_status = 'emissions-done'

    # coarsen emissions
    coarse_url = coarse_tile_template.format(tile_id=tile_id)
    coarse_mapper = fsspec.get_mapper(coarse_url)
    if not zarr_is_complete(coarse_mapper):
        ds = xr.open_zarr(mapper, consolidated=True)
        coarse_out = coarsen_emissions(ds, factor=COARSENING_FACTOR).chunk(coarse_chunks)
        print(coarse_out)
        print(coarse_out.nbytes / 1e9)
        coarse_mapper.clear()
        coarse_out.to_zarr(coarse_mapper, encoding=encoding, mode="w", consolidated=True)
        return_status = 'coarsen-done'

    return (return_status, url, coarse_url)


def combine_all_tiles(urls):
    print('combining all tiles')
    mapper = fsspec.get_mapper(coarse_full_template)

    if not zarr_is_complete(mapper):
        mapper.clear()
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            list_all_coarsened = [xr.open_zarr(url, consolidated=True) for url in urls]
            ds = xr.combine_by_coords(
                list_all_coarsened, compat="override", coords="minimal", combine_attrs="override"
            ).chunk(coarse_chunks)
            ds.attrs.update(get_cf_global_attrs())
            print(ds)
            print(ds.nbytes / 1e9)

        ds.to_zarr(mapper, consolidated=True)


def rollup_shapes():
    print('rollup_shapes')

    ds = xr.open_zarr(coarse_full_template, consolidated=True)
    shapes_df = geopandas.read_file(shapes_file)

    shapes_df['numbers'] = np.arange(len(shapes_df))
    mask = regionmask.mask_geopandas(shapes_df, ds.lon, ds.lat, numbers='numbers')

    # this will trigger dask compute
    df = ds['emissions'].groupby(mask).sum().to_pandas()

    # cleanup dataframe
    names = shapes_df['alpha3']
    columns = {k: names[int(k)] for k in df.columns}
    df = df.rename(columns=columns)

    # convert to teragrams
    df = df / 1e9

    # package in climate trace format
    df_out = df.stack().reset_index()
    df_out = df_out.sort_values(by=['region', 'year']).reset_index(drop=True)

    df_out['begin_date'] = pd.to_datetime(df_out.year, format='%Y')
    df_out['end_date'] = pd.to_datetime(df_out.year + 1, format='%Y')

    df_out = df_out.drop(columns=['year']).rename(columns={0: 'tCO2eq', 'region': 'iso3_country'})
    df_out = df_out[['iso3_country', 'begin_date', 'end_date', 'tCO2eq']]

    # write out
    df_out.to_csv(rollup_template, index=False)
    print(f'writing data to {rollup_template}')


def main():
    with Client(threads_per_worker=1, n_workers=16) as client:
        print(client)
        print(client.dashboard_link)

        urls = []
        coarse_urls = []
        for tile in tiles:
            result = process_one_tile(tile)
            urls.append(result[1])
            coarse_urls.append(result[2])
            if result[0] != 'skipped':
                client.restart()

        combine_all_tiles(coarse_urls)

        rollup_shapes()


if __name__ == "__main__":
    main()
