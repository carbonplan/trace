import dask
import fsspec
import numcodecs
import pandas as pd
import xarray as xr

from carbonplan_trace.metadata import get_cf_global_attrs
from carbonplan_trace.tiles import tiles
from carbonplan_trace.utils import zarr_is_complete
from carbonplan_trace.v0.core import coarsen_emissions
from carbonplan_trace.v0.data import cat
from carbonplan_trace.v0.data.load import _preprocess
from carbonplan_trace.v1 import utils


def open_biomass_tile(tile_id, version, y0=2014, y1=2021):
    input_tile_fn = f"s3://carbonplan-climatetrace/{version}/results/tiles/{tile_id}.zarr"
    ds = xr.open_zarr(input_tile_fn)

    # assign correct time coords
    time_coords = {'time': pd.date_range(f'{y0}-07-01', f'{y1}-07-01', freq='A')}
    ds = ds.assign_coords(time_coords)

    # use igbp land cover for masking
    lat, lon = utils.get_lat_lon_tags_from_tile_path(tile_id)
    bounding_box = utils.parse_bounding_box_from_lat_lon_tags(lat, lon)
    igbp = utils.open_global_igbp_data(lat_lon_box=bounding_box)
    land_mask = (igbp.igbp > 0).any(dim='year')
    ds['land_mask'] = utils.find_matching_records(data=land_mask, lats=ds.lat, lons=ds.lon)
    ds = ds.where(ds.land_mask)

    return ds


def open_hansen_biomass_tile(tile_id, version):
    """
    Open single tile from the Hansen 2020 dataset and then
    massage it into a format for use by the rest of the routines.

    Parameters
    ----------
    tile_id : str
        The latitude/longitude of the northwest corner of the tile (e.g. 50N_130W)
    version : str
        Spurilous param to conform with other functions

    Returns
    -------
    ds : xarray.Dataset
    """

    ds = xr.Dataset()
    lat, lon = tile_id.split('_')

    # Hansen biomass
    ds["agb"] = cat.gfw_biomass(lat=lat, lon=lon).to_dask().pipe(_preprocess).astype("float32")

    return ds


def coarsen_biomass_one_tile(
    tile_id,
    get_biomass_ds_func,
    output_template,
    variables,
    version,
    skip_existing=True,
    coarsening_factor=100,
    coarse_chunks={"lat": 400, "lon": 400},
):
    """
    Given lat and lon to select a region, coarsen biomass by averaging

    Parameters
    ----------
    tile_id : str
        String specifying which 10x10-degree tile to process, e.g. `50N_120W`
    get_biomass_ds_func : func
        A function that takes in two required params: tile_id and version, and returns the biomass ds to be coarsened
    output_template : str
        File name template for the output file, to be formatted with tile id
    variables : list
        Variables to be saved
    version : str
        Version of biomass data to process

    Returns
    -------
    url : string
        Url where a processed tile is located
    """
    print(tile_id)
    return_status = 'skipped'

    # mappers
    coarse_mapper = fsspec.get_mapper(output_template.format(tile_id=tile_id))

    # coarsen tile
    checks = [f'{v}/0.0.0' for v in variables]
    if not (skip_existing and zarr_is_complete(coarse_mapper, check=checks)):
        print('getting biomass')
        ds = get_biomass_ds_func(tile_id=tile_id, version=version)
        print('coarsen')
        coarse_out = coarsen_emissions(
            ds[variables], factor=coarsening_factor, mask_var=variables[0], method='mean'
        ).chunk(coarse_chunks)
        coarse_out.attrs.update(get_cf_global_attrs())

        coarse_mapper.clear()
        encoding = {"compressor": numcodecs.Blosc()}
        print('writing')
        coarse_out[variables].to_zarr(
            coarse_mapper,
            encoding={var: encoding for var in variables},
            mode="w",
            consolidated=True,
        )
        return_status = 'coarsen-done'

    return (return_status,)


def combine_all_tiles(
    input_tile_template,
    output_global_fn,
    variables,
    skip_existing=True,
    coarse_chunks={"lat": 400, "lon": 400},
):
    """
    Combine all tiles specified with input tile template into a global output file

    Parameters
    ----------
    input_tile_template : str
        File name template for the input tile file, to be formatted with tile id
    output_global_fn: str
        File name of output file
    variables : list
        Variables to be saved
    """
    print('combining all tiles')
    mapper = fsspec.get_mapper(output_global_fn)
    checks = [f'{v}/0.0.0' for v in variables]

    if not (skip_existing and zarr_is_complete(mapper, check=checks)):
        mapper.clear()
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            list_all_coarsened = []
            for tile_id in tiles:
                sub = xr.open_zarr(input_tile_template.format(tile_id=tile_id), consolidated=True)
                sub = sub.assign_coords(lat=sub.lat.round(6), lon=sub.lon.round(6))
                list_all_coarsened.append(sub)
            ds = xr.combine_by_coords(
                list_all_coarsened,
                compat="override",
                coords="minimal",
                combine_attrs="override",
            ).chunk(coarse_chunks)
            ds.attrs.update(get_cf_global_attrs())
            print(ds)
            print(ds.nbytes / 1e9)

        encoding = {"compressor": numcodecs.Blosc()}
        ds.to_zarr(mapper, encoding={var: encoding for var in variables}, consolidated=True)

    return 'done'
