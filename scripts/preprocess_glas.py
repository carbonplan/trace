import os
from collections import defaultdict

import dask
import fsspec
import xarray as xr
import zarr
from dask.distributed import Client
from gcsfs import GCSFileSystem

from carbonplan_trace.v1.glas_extract import extract_GLAH01_data, extract_GLAH14_data

fs = GCSFileSystem()

skip_existing = True
chunksize = 20000

drop_keys = {
    'GLAH01': ['rec_bin', 'shot_number', 'tx_bin'],
    'GLAH14': ['n_gaussian_peaks', 'shot_number'],
}


def get_mapper(uri):
    key = os.path.splitext(os.path.split(uri)[-1])[0]
    muri = f'gs://carbonplan-scratch/glas-zarr-cache/{key}.zarr'
    mapper = fsspec.get_mapper(muri)
    return mapper


@dask.delayed
def process_01(uri):
    mapper = get_mapper(uri)
    if skip_existing and '.zmetadata' in mapper:
        return ('skipped', uri)

    try:
        with dask.config.set(scheduler='single-threaded'):
            with fs.open(uri) as f:
                ds = extract_GLAH01_data(f).chunk(-1)
            ds.to_zarr(mapper, mode='w', consolidated=True)
            del ds
        return ('converted', uri)
    except:
        return ('failed', uri)


@dask.delayed
def process_14(uri):
    mapper = get_mapper(uri)

    if skip_existing and '.zmetadata' in mapper:
        return ('skipped', uri)

    try:
        with dask.config.set(scheduler='single-threaded'):
            with fs.open(uri) as f:
                ds = extract_GLAH14_data(f).chunk(-1)
            ds.to_zarr(mapper, mode='w', consolidated=True)
            del ds
        return ('converted', uri)
    except:
        return ('failed', uri)


process_funcs = {'GLAH01': process_01, 'GLAH14': process_14}


def granules_h5_to_zarr(products):
    # List all files in raw folder
    files = fs.ls('carbonplan-climatetrace/inputs/glas-raw/')
    files.sort()

    tasks = defaultdict(list)

    for uri in files:
        # skip xml files
        if uri.endswith('xml'):
            continue

        for p in products:
            if p in uri:
                tasks[p].append(process_funcs[p](uri))

    tasks = dict(tasks)
    results = dask.compute(tasks)[0]

    # unpack results from compute
    out = {}
    for p in products:
        out[p] = defaultdict(list)
        for k, v in results[p]:
            out[p][k].append(v)

    return out


@dask.delayed
def lazy_open(uri):
    ds = xr.open_zarr(get_mapper(uri), consolidated=True)
    return ds


def main(products=['GLAH01', 'GLAH14']):
    # convert hdf5 granules to zarr
    print('granules_h5_to_zarr')
    results = granules_h5_to_zarr(products)

    for p, presults in results.items():
        n_failed = len(presults['failed'])
        print(f'{n_failed} failed')
        uris = presults['skipped'] + presults['converted']
        n_files = len(uris)
        print(f'processing {n_files} files')

        print('open xarray datasets')
        ds_list = dask.compute([lazy_open(uri) for uri in uris])[0]

        print('concat list of datasets')
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ds = xr.concat(ds_list, dim='record_index').chunk({'record_index': chunksize})
            for k in ds:
                _ = ds[k].encoding.pop('chunks', None)

        # write
        print(f'writing {p}')
        print(ds)
        print(f'ds.nbytes: {ds.nbytes / 1e9}')
        mapper = fsspec.get_mapper(f'gs://carbonplan-climatetrace/intermediates/{p.lower()}.zarr')
        mapper.clear()

        # print('writing zarr dataset')
        ds.to_zarr(mapper, compute=False, mode='w')
        stepsize = chunksize * 10
        recs = ds.dims['record_index']
        print('writing zarr dataset chunks')
        for left, right in zip(
            range(0, recs, stepsize), range(stepsize, recs + stepsize, stepsize)
        ):
            s = slice(left, right)
            print(s, flush=True)
            ds.isel(record_index=s).drop(drop_keys[p]).to_zarr(mapper, region={'record_index': s})

        zarr.consolidate_metadata(mapper)


if __name__ == '__main__':

    client = Client(n_workers=12)
    print(client)
    print(client.dashboard_link)

    main(products=['GLAH14'])
