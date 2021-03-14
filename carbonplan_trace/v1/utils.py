import fsspec
from pyproj import Transformer


def save_to_zarr(ds, url, list_of_variables=None, mode='w'):
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

    ds[list_of_variables].to_zarr(mapper, mode=mode)


def get_transformer(p1=4326, p2=32610):
    """
    default p1 p2 transforms from lat/lon to Landsat coordinates
    """
    return Transformer.from_crs(4326, 32610)


def get_x_from_latlon(lat, lon, transformer):
    x, y = transformer.transform(lat, lon)
    return x


def get_y_from_latlon(lat, lon, transformer):
    x, y = transformer.transform(lat, lon)
    return y


def convert_long3_to_long1(long3):
    # see https://confluence.ecmwf.int/pages/viewpage.action?pageId=149337515
    long1 = (long3 + 180) % 360 - 180
    return long1
