import fsspec


def save_to_zarr(ds, url, list_of_variables=None):
    """
    Avoid chunking errors while saving a dataset to zarr file
    list_of_variables is a list of variables to store, everything else will be dropped
    if None then the dataset will be stored as is
    """
    mapper = fsspec.get_mapper(url)

    if not list_of_variables:
        list_of_variables = list(ds.keys())

    for v in list_of_variables:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']

    ds[list_of_variables].to_zarr(mapper, mode='w')
