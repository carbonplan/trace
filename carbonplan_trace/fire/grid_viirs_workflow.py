#!/usr/bin/env python3

import warnings

import fsspec
import geopandas
import numcodecs
import numpy as np
import rasterio
import xarray as xr
from rasterio.features import rasterize
from tenacity import retry

from carbonplan_trace.metadata import get_cf_global_attrs
from carbonplan_trace.tiles import tiles
from carbonplan_trace.utils import zarr_is_complete
from carbonplan_trace.v0.data import cat

out_template = (
    's3://carbonplan-climatetrace/intermediate/viirs_fire/tiles/{resolution}m/{tile_id}_{year}.zarr'
)
chunks = {"lat": 4000, "lon": 4000}
coarse_chunks = {"lat": 400, "lon": 400}
encoding = {"burned_area": {"compressor": numcodecs.Blosc()}}
resolution = 30
COARSENING_FACTOR = 100


def calc_buffer_distance(lats, buffer_m=375):
    deg_at_eq_m = 110574.2727

    buffer = buffer_m / (deg_at_eq_m * np.cos(np.deg2rad(lats)))

    return buffer


def rasterize_geom(geoms, transform, shape):

    r = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=shape,
        transform=transform,
        fill=0,
        merge_alg=rasterio.enums.MergeAlg.replace,
        all_touched=True,
        dtype=rasterio.uint8,
    )
    return r


def tile_id_to_slices(tile_id, width=10):
    lat_str, lon_str = tile_id.split('_')

    if 'N' in lat_str:
        lat = int(lat_str[:-1])
    elif 'S' in lat_str:
        lat = -1 * int(lat_str[:-1])
    else:
        raise ValueError

    if 'E' in lon_str:
        lon = int(lon_str[:-1])
    elif 'W' in lon_str:
        lon = -1 * int(lon_str[:-1])
    else:
        raise ValueError

    lon_slice = slice(lon, lon + width)
    lat_slice = slice(lat, lat - width)

    return lon_slice, lat_slice


@retry
def open_target_grid(tile_id):
    # open template data
    lat, lon = tile_id.split('_')
    target_da = (
        cat.hansen_change(lat=lat, lon=lon)
        .to_dask()
        .squeeze(drop=True)
        .rename({'x': 'lon', 'y': 'lat'})
    )
    return target_da


def process_one_year(year):

    print(year)
    gdf = geopandas.read_parquet(
        f's3://carbonplan-climatetrace/inputs/processed/viirs/{year}.parquet'
    )

    # filter gdf
    gdf = gdf[gdf['CONFIDENCE'].isin(['n', 'h'])]

    # buffer points
    # This needs a review
    buffer = calc_buffer_distance(gdf['LATITUDE'])
    print('buffering points')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignoring UserWarning since we are using a latitude adjusted buffer
        gdf = gdf.set_geometry(gdf.buffer(buffer))

    for tile_id in tiles:
        print(tile_id)

        # output mappers
        uri = out_template.format(resolution=resolution, tile_id=tile_id, year=year)
        mapper = fsspec.get_mapper(uri)

        coarse_uri = out_template.format(
            resolution=resolution * COARSENING_FACTOR, tile_id=tile_id, year=year
        )
        coarse_mapper = fsspec.get_mapper(coarse_uri)

        if zarr_is_complete(mapper) and zarr_is_complete(coarse_mapper):
            continue

        lon_slice, lat_slice = tile_id_to_slices(tile_id)
        print(lon_slice, lat_slice)
        gdf_box = gdf.cx[lon_slice, lat_slice]
        print(f'found {len(gdf_box)} fire pixels in box')

        target_da = open_target_grid(tile_id)

        # rasterize geometries
        print('rasterizing')
        # hard coding shape/transform because it turns out the hansen data can't be trusted to provide an accurate affine
        shape = (40000, 40000)
        transform = rasterio.Affine(0.00025, 0.0, lon_slice.start, 0.0, -0.00025, lat_slice.start)
        print(shape, repr(transform))
        if len(gdf_box) > 0:
            mask = rasterize_geom(gdf_box.geometry.values, transform, shape)
        else:
            mask = np.zeros(shape, dtype=np.uint8)
        mask_da = xr.DataArray(mask, dims=target_da.dims, coords=target_da.coords)
        print('done rasterizing, found %d fire pixels' % mask_da.sum().item())

        # package dataset
        out = mask_da.to_dataset(name='burned_area').chunk(chunks)
        out.attrs.update(get_cf_global_attrs())

        # write full res
        print(uri)
        out.to_zarr(mapper, encoding=encoding, mode="w", consolidated=True)

        # write coarsened version
        out_coarse = (
            out.coarsen(lon=COARSENING_FACTOR, lat=COARSENING_FACTOR).mean().chunk(coarse_chunks)
        )
        print(coarse_uri)
        out_coarse.to_zarr(coarse_mapper, encoding=encoding, mode="w", consolidated=True)


if __name__ == "__main__":
    for year in range(2020, 2011, -1):
        process_one_year(year)
