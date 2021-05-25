import dask
import fsspec
import numcodecs
from dask.distributed import Client
from prefect import Flow, task
from prefect.executors import DaskExecutor

from carbonplan_trace.metadata import get_cf_global_attrs
from carbonplan_trace.tiles import tiles
from carbonplan_trace.v0.core import calc_emissions
from carbonplan_trace.v0.data.load import open_hansen_change_tile

template = "s3://carbonplan-climatetrace/v0.1/tiles/30m/{tile_id}.zarr"
chunks = {"lat": 4000, "lon": 4000, "year": 2}


@task
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

    with dask.config.set(scheduler="threads"):
        url = template.format(tile_id=tile_id)

        encoding = {"emissions": {"compressor": numcodecs.Blosc()}}

        mapper = fsspec.get_mapper(url)

        lat, lon = tile_id.split('_')
        ds = open_hansen_change_tile(lat, lon)

        out = calc_emissions(ds).to_dataset(name='emissions').chunk(chunks)
        out.attrs.update(get_cf_global_attrs())

        print(out)
        print(out.nbytes / 1e9)
        # TODO: add metadata to emissions variable
        out.to_zarr(mapper, encoding=encoding, mode="w", consolidated=True)

    return url


def main():
    with Client(n_workers=1, threads_per_worker=1) as client:
        print(client)
        print(client.dashboard_link)

        executor = DaskExecutor(address=client.cluster.scheduler_address)
        with Flow('v0-emissions-workflow') as flow:
            mapped_results = process_one_tile.map(tiles[:4])
        flow.run(executor=executor)

    print(mapped_results)


if __name__ == "__main__":
    main()
