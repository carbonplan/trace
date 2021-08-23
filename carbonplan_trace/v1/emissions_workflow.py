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
from carbonplan_trace.v0.core import coarsen_emissions

from ..constants import EMISSIONS_FACTORS, SQM_PER_HA, TC02_PER_TC, TC_PER_TBM, R

skip_existing = True
tile_template = "s3://carbonplan-climatetrace/v1/results/tiles/30m/{tile_id}_{kind}.zarr"
coarse_tile_template = "s3://carbonplan-climatetrace/v1/tiles/3000m/{tile_id}_{kind}.zarr"
coarse_full_template = "s3://carbonplan-climatetrace/v0.4/global/3000m/raster_{kind}.zarr"
shapes_file = 's3://carbonplan-climatetrace/inputs/shapes/countries.shp'
rollup_template = 's3://carbonplan-climatetrace/v0.4/country_rollups_{var}.csv'
chunks = {"lat": 4000, "lon": 4000, "year": 2}
coarse_chunks = {"lat": 400, "lon": 400, "year": -1}
years = (2001, 2020)
COARSENING_FACTOR = 100
tot_encoding = {"flux": {"compressor": numcodecs.Blosc()}}
split_encoding = {
    "emissions_from_clearing": {"compressor": numcodecs.Blosc()},
    "emissions_from_fire": {"compressor": numcodecs.Blosc()},
}


def open_fire_mask(tile_id, resolution=30, y0=2014, y1=2020):
    fire_template = 's3://carbonplan-climatetrace/intermediate/viirs_fire/tiles/{resolution}m/{tile_id}_{year}.zarr'

    years = xr.DataArray(range(y0, y1 + 1), dims=("year",), name="year")
    fires = xr.concat(
        [
            xr.open_zarr(
                fire_template.format(resolution=resolution, tile_id=tile_id, year=int(year)),
                consolidated=True,
            )['burned_area']
            for year in years
        ],
        dim=years,
    )

    return fires


def calc_biomass_change(ds):
    # diff by one year, dropping the first year since it will be all nulls
    return (ds.shift(time=1) - ds).isel(time=slice(1, None))


def convert_to_emissions(ds):
    # emissions factor
    ds = fire_emissions_factor_conversion(ds)
    for variable in ['sinks', 'emissions_from_clearing']:
        ds[variable] = ds[variable] * TC_PER_TBM * TC02_PER_TC
    return ds


def fire_emissions_factor_conversion(ds):
    # load tropics mask (different emisisons factor)
    ds = load.tropics(ds)

    min_lat = data.y.min().values
    max_lat = data.y.max().values
    min_lon = data.x.min().values
    max_lon = data.x.max().values

    lat_lon_box = min_lat, max_lat, min_lon, max_lon
    # load forest type mask
    igbp = utils.open_global_igbp_data(lat_lon_box=lat_lon_box)

    savanna_mask = igbp.igbp.isin([8, 9]).astype('int')

    for year in [2019, 2020]:
        new_savanna_mask = savanna_mask.sel(year=2018).assign_coords({'year': [year]})
        savanna_mask = xr.concat([savanna_mask, new_savanna_mask], dim='year')

    ds['is_savanna'] = utils.find_matching_records(
        data=savanna_mask, lats=ds.y, lons=ds.x, year=ds.time
    )

    ds['savanna_fires'] = ds['emissions_from_fires'].where(ds['is_savanna'] == 1)
    # subtract out savanna fires
    ds['emissions_from_fires'] = ds['emissions_from_fires'] - ds['savanna_fires']

    ds['tropical_forest_fires'] = ds['emissions_from_fires'].where(
        ds['is_tropics'] == 1 & ds['is_savanna'] == 0
    )

    # subtract out tropical forest fires
    ds['extratropical_forest_fires'] = ds['emissions_from_fires'] - ds['tropical_forest_fires']

    del ds['emissions_from_fires']

    for (fire_type, emissions_factor) in EMISSIONS_FACTORS.items():
        ds[fire_type] = ds[fire_type] * emissions_factor

    return ds


def process_one_tile(tile_id):
    """
    Given lat and lon to select a region, calculate the corresponding emissions for each year

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

    # mappers
    tot_mapper = fsspec.get_mapper(tile_template.format(tile_id=tile_id, kind='tot'))
    split_mapper = fsspec.get_mapper(tile_template.format(tile_id=tile_id, kind='split'))
    coarse_tot_mapper = fsspec.get_mapper(coarse_tile_template.format(tile_id=tile_id, kind='tot'))
    coarse_split_mapper = fsspec.get_mapper(
        coarse_tile_template.format(tile_id=tile_id, kind='split')
    )

    # calc emissions
    if not (skip_existing and zarr_is_complete(tot_mapper) and zarr_is_complete(split_mapper)):

        # read data 
        fire_da = open_fire_mask(tile_id).fillna(0)
        carbon_pools = xr.open_zarr(f's3://carbonplan-climatetrace/v1/results/tiles/{tile_id}.zarr')
        
        # calculate fluxes         
        flux = calc_biomass_change(ds=carbon_pools)
        sources = flux.clip(min=0)
        sinks = flux.clip(max=0)
        # write total emissions

        # split out emissions from fire/clearing
        out = xr.Dataset()

        # emissions occuring on the year of a fire or the year after a fire in the same pixel
        # are marked as emissions from fire. this is consistent with the methods in Harris et al 2021.
        # note that we are limited by the start of the dataset and will miss the fires from years[0] - 1
        fire_attribution = (fire_da + fire_da.shift(year=1, fill_value=0)).astype(bool)
        sources, fire_attribution = xr.align(sources, fire_attribution, join='inner')

        # this needs to be changed because sources is a dataset with four data arrays (AGB, BGB, deadwood, litter)
        # we can't assign it to be a dataarray in the out dataset 
        out['emissions_from_fire'] = sources.where(fire_attribution, other=0)
        out['emissions_from_clearing'] = sources - out['emissions_from_fire']
        out['sinks'] = sinks
        out = convert_to_emissions(out)

        out.attrs.update(get_cf_global_attrs())

        # TODO: add metadata to emissions variable
        out.to_zarr(split_mapper, encoding=split_encoding, mode="w", consolidated=True)

        sources.attrs.update(get_cf_global_attrs())
        sources.to_zarr(tot_mapper, encoding=tot_encoding, mode="w", consolidated=True)

        return_status = 'emissions-done'

    # coarsen emissions
    for in_mapper, out_mapper, encoding in [
        (tot_mapper, coarse_tot_mapper, tot_encoding),
        (split_mapper, coarse_split_mapper, split_encoding),
    ]:

        if not (skip_existing and zarr_is_complete(out_mapper)):
            ds = xr.open_zarr(in_mapper, consolidated=True)

            mask_var = list(encoding.keys())[0]
            coarse_out = coarsen_emissions(ds, factor=COARSENING_FACTOR, mask_var=mask_var).chunk(
                coarse_chunks
            )
            coarse_out.attrs.update(get_cf_global_attrs())

            out_mapper.clear()
            coarse_out.to_zarr(out_mapper, encoding=encoding, mode="w", consolidated=True)
            return_status = 'coarsen-done'

    return (return_status,)


def combine_all_tiles():
    print('combining all tiles')

    for kind, encoding in [('tot', tot_encoding), ('split', split_encoding)]:

        mapper = fsspec.get_mapper(coarse_full_template.format(kind=kind))

        if not (skip_existing and zarr_is_complete(mapper)):
            mapper.clear()
            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                list_all_coarsened = [
                    xr.open_zarr(
                        coarse_tile_template.format(tile_id=tile_id, kind=kind), consolidated=True
                    )
                    for tile_id in tiles
                ]
                ds = xr.combine_by_coords(
                    list_all_coarsened,
                    compat="override",
                    coords="minimal",
                    combine_attrs="override",
                ).chunk(coarse_chunks)
                ds.attrs.update(get_cf_global_attrs())
                print(ds)
                print(ds.nbytes / 1e9)

            ds.to_zarr(mapper, encoding=encoding, consolidated=True)


def rollup_shapes():
    print('rollup_shapes')

    shapes_df = geopandas.read_file(shapes_file)
    shapes_df['numbers'] = np.arange(len(shapes_df))

    for kind, var_names in [
        ('tot', ['emissions']),
        ('split', ['emissions_from_clearing', 'emissions_from_fire']),
    ]:

        ds = xr.open_zarr(coarse_full_template.format(kind=kind), consolidated=True)

        mask = regionmask.mask_geopandas(shapes_df, ds['lon'], ds['lat'], numbers='numbers')

        for var in var_names:

            # this will trigger dask compute
            df = ds[var].groupby(mask).sum().to_pandas()

            # cleanup dataframe
            names = shapes_df['alpha3']
            columns = {k: names[int(k)] for k in df.columns}
            df = df.rename(columns=columns)

            # package in climate trace format
            df_out = df.stack().reset_index()
            df_out = df_out.sort_values(by=['region', 'year']).reset_index(drop=True)

            df_out['begin_date'] = pd.to_datetime(df_out.year, format='%Y')
            df_out['end_date'] = pd.to_datetime(df_out.year + 1, format='%Y')

            df_out = df_out.drop(columns=['year']).rename(
                columns={0: 'tCO2eq', 'region': 'iso3_country'}
            )
            df_out = df_out[['iso3_country', 'begin_date', 'end_date', 'tCO2eq']]

            # write out
            uri = rollup_template.format(var=var)
            df_out.to_csv(uri, index=False)
            print(f'writing data to {uri}')


def main():
    with Client(threads_per_worker=1, n_workers=12) as client:
        print(client)
        print(client.dashboard_link)

        for tile in tiles:
            result = process_one_tile(tile)
            if result[0] != 'skipped':
                client.restart()

        combine_all_tiles()
        rollup_shapes()


if __name__ == "__main__":
    main()
