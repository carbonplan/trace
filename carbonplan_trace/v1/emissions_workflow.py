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
from carbonplan_trace.v1 import load, utils

from ..constants import EMISSIONS_FACTORS, TC02_PER_TC, TC_PER_TBM_IPCC

skip_existing = True
tile_template = "s3://carbonplan-climatetrace/v1.2/results/tiles/30m/{tile_id}_{kind}.zarr"
coarse_tile_template = "s3://carbonplan-climatetrace/v1.2/results/tiles/3000m/{tile_id}_{kind}.zarr"
coarse_full_template = "s3://carbonplan-climatetrace/v1.2/results/global/3000m/raster_{kind}.zarr"
shapes_file = 's3://carbonplan-climatetrace/inputs/shapes/countries.shp'
rollup_template = 's3://carbonplan-climatetrace/v1.2/country_rollups_{var}.csv'
chunks = {"lat": 4000, "lon": 4000, "year": 2}
coarse_chunks = {"lat": 400, "lon": 400, "year": -1}
years = (2001, 2020)
COARSENING_FACTOR = 100
tot_encoding = {"flux": {"compressor": numcodecs.Blosc()}}
split_encoding = {
    "sinks": {"compressor": numcodecs.Blosc()},
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
    ds = ds.rename({'time': 'year'}).assign_coords({'year': np.arange(2014, 2021)})
    # diff by one year, dropping the first year since it will be all nulls
    return (ds.shift(year=1) - ds).isel(year=slice(1, None))


def convert_to_emissions(ds, emissions_from_clearing, sinks):
    # emissions factor
    ds = fire_emissions_factor_conversion(ds)
    for emissions_ds in [emissions_from_clearing, sinks]:
        for carbon_pool, biomass_carbon_conversion_factor in TC_PER_TBM_IPCC.items():
            emissions_ds[carbon_pool] *= biomass_carbon_conversion_factor * TC02_PER_TC

    ds['emissions_from_clearing'] = emissions_from_clearing.to_array(dim='var').sum(dim='var')
    ds['sinks'] = emissions_from_clearing.to_array(dim='var').sum(dim='var')
    return ds


def fire_emissions_factor_conversion(ds):
    # load tropics mask (different emisisons factor)
    # just select one slice to make it 2d template
    is_tropics = load.tropics(
        ds, template_var='emissions_from_fire', chunks_dict={'y': 4000, 'x': 4000}
    )
    min_lat = ds.y.min().values
    max_lat = ds.y.max().values
    min_lon = ds.x.min().values
    max_lon = ds.x.max().values

    lat_lon_box = min_lat, max_lat, min_lon, max_lon
    # load forest type mask

    igbp = utils.open_global_igbp_data(lat_lon_box=lat_lon_box)
    savanna_mask = igbp.igbp.isin([8, 9]).astype('int')

    new_years = xr.DataArray(list(range(2015, 2021)), dims='year')
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        savanna_mask = savanna_mask.reindex(year=new_years).ffill('year')

    savanna_mask = savanna_mask.to_dataset()
    is_savanna = utils.find_matching_records(
        data=savanna_mask, lats=ds.y, lons=ds.x, years=ds.year
    )['igbp']
    is_savanna = is_savanna.chunk({'year': 1, 'y': 4000, 'x': 4000}).astype(bool)

    savanna_fire = ds['emissions_from_fire'].where(is_savanna).fillna(0)
    # subtract out savanna fires
    ds['emissions_from_fire'] = ds['emissions_from_fire'] - savanna_fire

    tropical_forest_fire = (
        ds['emissions_from_fire'].where((is_tropics == 1) & (is_savanna == 0)).fillna(0)
    )

    # subtract out tropical forest fires
    extratropical_forest_fire = ds['emissions_from_fire'] - tropical_forest_fire

    extratropical_forest_fire *= EMISSIONS_FACTORS['extratropical_forest_fire']
    tropical_forest_fire *= EMISSIONS_FACTORS['tropical_forest_fire']
    savanna_fire *= EMISSIONS_FACTORS['savanna_fire']

    ds['emissions_from_fire'] = tropical_forest_fire + extratropical_forest_fire + savanna_fire
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
    return_status = 'skipped'

    # mapper
    split_mapper = fsspec.get_mapper(tile_template.format(tile_id=tile_id, kind='split'))
    # this will be a more rigorous check than just for the metadata- will also check for a chunk being written
    checks = [f'{v}/0.0.0' for v in split_encoding.keys()]
    checks.append('.zmetadata')
    # calc emissions
    if not (skip_existing and zarr_is_complete(split_mapper)):
        print(tile_id)
        # read data
        fire_da = open_fire_mask(tile_id).fillna(0)
        carbon_pools = xr.open_zarr(
            f's3://carbonplan-climatetrace/v1.2/results/tiles/{tile_id}.zarr'
        )

        # calculate fluxes
        flux = calc_biomass_change(ds=carbon_pools)
        sources = flux.clip(min=0)  # .compute()
        sinks = flux.clip(max=0)  # .compute()
        # del flux
        # write total emissions

        # split out emissions from fire/clearing
        out = xr.Dataset()

        # emissions occuring on the year of a fire or the year after a fire in the same pixel
        # are marked as emissions from fire. this is consistent with the methods in Harris et al 2021.
        # note that we are limited by the start of the dataset and will miss the fires from years[0] - 1
        fire_attribution = (fire_da + fire_da.shift(year=1, fill_value=0)).astype(bool)
        # # # corrects for tiny offsets in the coordinate values
        fire_attribution['lat'] = fire_attribution.lat.round(6)
        fire_attribution['lon'] = fire_attribution.lon.round(6)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # switch lats so they go from lower to higher and then chunk
            fire_attribution = fire_attribution.reindex(lat=list(reversed(fire_attribution.lat)))
        fire_attribution = fire_attribution.chunk({'year': 1, 'lat': 4000, 'lon': 4000})
        # create inverse fire mask
        # create inverse fire mask
        clearing_attribution = (1 - fire_attribution.astype(int)).astype(bool)
        sources['lat'] = fire_attribution.lat
        sources['lon'] = fire_attribution.lon
        # the align statement should just restrict to the overlapping time dimension since we've
        # already aligned lat/lon above
        sources, fire_attribution = xr.align(sources, fire_attribution, join='inner')
        sources, clearing_attribution = xr.align(sources, clearing_attribution, join='inner')
        # # this needs to be changed because sources is a dataset with four data arrays (AGB, BGB, deadwood, litter)
        # # we can't assign it to be a dataarray in the out dataset
        # # include different carbon pools per Table S5 in Harris et al. (2020)
        out['emissions_from_fire'] = (sources['AGB'] + sources['BGB']).where(
            fire_attribution, other=0
        )
        emissions_from_clearing = sources.where(clearing_attribution, other=0)
        out = out.rename({'lat': 'y', 'lon': 'x'})
        emissions_from_clearing = emissions_from_clearing.rename({'lat': 'y', 'lon': 'x'})
        sinks = sinks.rename({'lat': 'y', 'lon': 'x'})
        out = convert_to_emissions(out, emissions_from_clearing, sinks)
        out = out.rename({'x': 'lon', 'y': 'lat'})
        out = out.chunk(chunks)
        out.attrs.update(get_cf_global_attrs())

        out.to_zarr(split_mapper, encoding=split_encoding, mode="w", consolidated=True)

        return_status = 'emissions-done'
    else:
        return_status = 'skipped'

    return (return_status,)


def coarsen_tile(tile_id):
    split_mapper = fsspec.get_mapper(tile_template.format(tile_id=tile_id, kind='split'))
    coarse_split_mapper = fsspec.get_mapper(
        coarse_tile_template.format(tile_id=tile_id, kind='split')
    )
    for in_mapper, out_mapper, encoding in [
        (split_mapper, coarse_split_mapper, split_encoding),
    ]:

        if not (skip_existing and zarr_is_complete(out_mapper)):
            ds = utils.open_result_tile(
                tile_id, variable='emissions', version='v1.2', resolution='30m', apply_masks=True
            )

            coarse_out = coarsen_emissions(ds, factor=COARSENING_FACTOR, mask_var='sinks').chunk(
                coarse_chunks
            )
            # mask to land and where we had valid landsat data
            coarse_out = utils.apply_result_masks(tile_id, coarse_out)
            coarse_out.attrs.update(get_cf_global_attrs())

            out_mapper.clear()
            coarse_out = coarse_out.chunk(coarse_chunks)
            coarse_out.to_zarr(out_mapper, encoding=encoding, mode="w", consolidated=True)
            return_status = 'coarsen-done'
        else:
            return_status = 'skipped'
    return return_status


def combine_all_tiles(encoding_kinds=[('split', split_encoding)]):
    print('combining all tiles')

    for kind, encoding in encoding_kinds:

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

        for tile in tiles[0:]:
            result = process_one_tile(tile)
            if result[0] != 'skipped':
                client.restart()
        for tile in tiles:
            coarsen_tile(tile)
        combine_all_tiles(encoding_kinds=[('split', split_encoding)])
        rollup_shapes()


if __name__ == "__main__":
    main()
