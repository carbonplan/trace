import math
import os

import awswrangler as wr
import boto3
import fsspec
import numpy as np
import utm
import xarray as xr
from pyproj import Transformer
from s3fs import S3FileSystem


def save_to_zarr(ds, url, list_of_variables=None, mode='w', append_dim=None):
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

    ds[list_of_variables].to_zarr(mapper, mode=mode, append_dim=append_dim, consolidated=True)


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


def open_zarr_file(uri, file_system='s3'):
    if not uri.startswith(f'{file_system}://'):
        uri = f'{file_system}://{uri}'
    mapper = fsspec.get_mapper(uri)
    ds = xr.open_zarr(mapper)
    return ds


def open_glah14_data(do_convert_long3_to_long1=True):
    data = open_zarr_file("s3://carbonplan-climatetrace/intermediate/glah14.zarr")
    if do_convert_long3_to_long1:
        data["lon"] = convert_long3_to_long1(data.lon)
    return data


def open_glah01_data():
    fs = S3FileSystem()
    uris = [
        f's3://{f}'
        for f in fs.ls('s3://carbonplan-climatetrace/intermediate/glah01/')
        if not f.endswith('/')
    ]
    ds_list = [open_zarr_file(uri) for uri in uris]
    ds = xr.concat(ds_list, dim='record_index').chunk({'record_index': 500})
    for k in ds:
        _ = ds[k].encoding.pop('chunks', None)
    return ds


def open_and_combine_lat_lon_data(folder, tiles=None, lat_lon_box=None):
    """
    Load lat lon data stored as 10x10 degree tiles in folder
    If tiles is none, load all data available
    If no file is available, return None
    """
    fs = S3FileSystem()

    if not tiles:
        tiles = [
            os.path.splitext(os.path.split(path)[-1])[0]
            for path in fs.ls(folder)
            if not path.endswith('/')
        ]

    uris = [f'{folder}{tile}.zarr' for tile in tiles]
    ds_list = []
    for uri in uris:
        if fs.exists(uri):
            da = open_zarr_file(uri)
            if da.lat[0] > da.lat[-1]:
                da = da.reindex(lat=da.lat[::-1])
            if da.lon[0] > da.lon[-1]:
                da = da.reindex(lat=da.lon[::-1])

            if lat_lon_box is not None:
                [min_lat, max_lat, min_lon, max_lon] = lat_lon_box
                da = da.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

            if da.dims['lat'] > 0 and da.dims['lon'] > 0:
                ds_list.append(da)
    if len(ds_list) > 0:
        ds = xr.combine_by_coords(ds_list, combine_attrs="drop_conflicts").chunk(
            {'lat': 2000, 'lon': 2000}
        )
        return ds
    # print(f'No data available at {folder} for tiles {tiles}')
    return None


def open_srtm_data(tiles=None):
    """
    Load SRTM data stored as 10x10 degree tiles
    If tiles is none, load all data available
    """
    folder = 's3://carbonplan-climatetrace/intermediate/srtm/'
    ds = open_and_combine_lat_lon_data(folder, tiles)

    return ds


def open_ecoregion_data(tiles=None):
    """
    Load ecoregion data stored as 10x10 degree tiles
    If tiles is none, load all data available
    """
    folder = 's3://carbonplan-climatetrace/intermediate/ecoregions_mask/'

    return open_and_combine_lat_lon_data(folder, tiles)


def open_igbp_data(tiles=None, lat_lon_box=None):
    """
    Load igbp data stored as 10x10 degree tiles
    If tiles is none, load all data available
    """
    folder = 's3://carbonplan-climatetrace/intermediate/igbp/'

    return open_and_combine_lat_lon_data(folder, tiles, lat_lon_box=lat_lon_box)


def open_burned_area_data(tiles):
    """
    Load MODIS burned area data stored as 10x10 degree tiles
    If tiles is none, load all data available
    """
    folder = 's3://carbonplan-climatetrace/intermediate/modis_burned_area/'

    return open_and_combine_lat_lon_data(folder, tiles)


def write_parquet(df, out_path, access_key_id, secret_access_key):
    wr.s3.to_parquet(
        df=df,
        index=True,
        path=out_path,
        boto3_session=boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        ),
    )


def find_matching_records(data, lats, lons, years=None, dtype=None):
    """
    find records in data that is nearest to locations specified in lats and lons
    lat and lon must be coordinates in data (an xarray dataset/daraarray)
    """
    if dtype is not None:
        if years is not None:
            assert 'year' in data
            return (
                data.sel(lat=lats, lon=lons, year=years, method="nearest")
                .drop_vars(["lat", "lon", "year"])
                .astype(dtype)
            )

        return (
            data.sel(lat=lats, lon=lons, method="nearest").drop_vars(["lat", "lon"]).astype(dtype)
        )

    else:
        if years is not None:
            assert 'year' in data
            return data.sel(lat=lats, lon=lons, year=years, method="nearest").drop_vars(
                ["lat", "lon", "year"]
            )

        return data.sel(lat=lats, lon=lons, method="nearest").drop_vars(["lat", "lon"])


def get_lat_lon_tags_from_tile_path(tile_path):
    """
    tile_path may be the full path including gs://, folder names, and extension
    outputs a lat lon tag eg, (50N, 120W)
    """
    fn = os.path.splitext(os.path.split(tile_path)[-1])[0]
    lat, lon = fn.split('_')

    return lat, lon


def parse_bounding_box_from_lat_lon_tags(lat, lon):
    """
    lat lon strings denoting the upper left corner of a 10x10 degree box eg (50N, 120W)
    """
    # the tile name denotes the upper left corner of each tile
    if lat.endswith('N'):
        max_lat = float(lat[:-1])
    elif lat.endswith('S'):
        max_lat = -1 * float(lat[:-1])
    # each tile covers 10 degree x 10 degree
    min_lat = max_lat - 10

    if lon.endswith('E'):
        min_lon = float(lon[:-1])
    elif lon.endswith('W'):
        min_lon = -1 * float(lon[:-1])
    max_lon = min_lon + 10

    return min_lat, max_lat, min_lon, max_lon


def get_lat_lon_tags_from_bounding_box(max_lat, min_lon):
    lat_tag = str(abs(math.ceil(max_lat))).zfill(2)
    if max_lat >= 0:
        lat_tag += 'N'
    else:
        lat_tag += 'S'

    lon_tag = str(abs(math.floor(min_lon))).zfill(3)
    if min_lon >= 0:
        lon_tag += 'E'
    else:
        lon_tag += 'W'

    return lat_tag, lon_tag


def subset_data_for_bounding_box(data, min_lat, max_lat, min_lon, max_lon):
    """
    Return a subset of data within the bounding lat/lon box
    The function assumes that lat/lon are not coordinates in the data
    """
    sub = data.where(
        (data.lat > min_lat) & (data.lat <= max_lat) & (data.lon > min_lon) & (data.lon <= max_lon),
        drop=True,
    )
    return sub


def find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon):
    """
    return a list of 10x10 degree tile names covering the bounding box
    the tile names are in the format of {lat}_{lon} where lat, lon represent the upper left corner
    ocean tiles are removed
    """
    fs = S3FileSystem()
    folder = 's3://carbonplan-climatetrace/intermediate/ecoregions_mask/'
    available_tiles = [
        os.path.splitext(os.path.split(path)[-1])[0]
        for path in fs.ls(folder)
        if not path.endswith('/')
    ]

    step = 10
    lat_start = math.ceil(min_lat / step) * step
    lat_stop = math.ceil(max_lat / step) * step
    all_lat_tiles = np.arange(start=lat_start, stop=lat_stop + 1, step=step)
    if min_lat == lat_start:
        all_lat_tiles = all_lat_tiles[1:]

    lon_start = math.floor(min_lon / step) * step
    lon_stop = math.floor(max_lon / step) * step
    all_lon_tiles = np.arange(start=lon_start, stop=lon_stop + 1, step=step)
    if max_lon == lon_stop:
        all_lon_tiles = all_lon_tiles[:-1]

    out = []
    for lat in all_lat_tiles:
        for lon in all_lon_tiles:
            lat_tag, lon_tag = get_lat_lon_tags_from_bounding_box(lat, lon)
            fn = f'{lat_tag}_{lon_tag}'
            if fn in available_tiles:
                out.append(fn)

    return out


# create utm band letter / latitude dictionary
# latitude represents southern edge of letter band
BAND_NUMBERS = list(np.arange(-80, 80, 8))
BAND_NUMBERS.append(84)


def spans_utm_border(lats):
    '''
    find if a latitude range of a scene spans more than 1 band
    '''
    min_lat = np.min(np.array(lats))
    max_lat = np.max(np.array(lats))

    if ((BAND_NUMBERS < max_lat).astype(int) + (BAND_NUMBERS > min_lat).astype(int) != 1).sum():
        # this logic only evaluates if the lats span more than
        # one interval in the lat bands
        return True
    else:
        return False


def verify_projection(coords, projected, zone_number):
    '''
    Use UTM to project a provided lat/lon coordinate into
    x/y space and see if they match.
    If they do, grab a letter. If not, grab the other letter (and
    confirm that it also works?)
    '''
    tolerance = 2  # in meters - should really be within 0.5 meters

    for corner in ['UR', 'LL', 'UL', 'LR']:
        # test out for a given coordinate
        (test_x, test_y, calculated_zone_number, calculated_zone_letter) = utm.from_latlon(
            coords[corner][1], coords[corner][0], force_zone_number=zone_number
        )
        if coords[corner][1] < 0 and abs(test_y - projected[corner][1]) > tolerance:
            # this line is implemented in response to 
            # https://github.com/Turbo87/utm/blob/40eb34c86895bf3a5f97b5819b9da4b164151d3c/utm/conversion.py#L283-L284
            # without it some areas get a 10M difference in northing/y
            test_y -= 10000000
        # These will fail if the test latlon-->meters projection was off by more than
        # 2 meters from the values provided in the metadata
        try:
            assert abs(test_x - projected[corner][0]) < tolerance
            assert abs(test_y - projected[corner][1]) < tolerance
            return calculated_zone_letter
        except AssertionError:
            continue

    raise Exception('None of the UTM projections match with scene metadata')
