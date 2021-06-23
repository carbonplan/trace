import gc
import json

import boto3
import dask
import numpy as np
import rasterio as rio
import rioxarray  # for the extension to load
import utm
import xarray as xr
from rasterio.session import AWSSession
from s3fs import S3FileSystem

from .utils import spans_utm_border, verify_projection

# flake8: noqa


def test_credentials(
    aws_session,
    canary_file='s3://usgs-landsat/collection02/level-2/standard/'
    + 'tm/2003/044/029/LT05_L2SP_044029_20030827_20200904_02_T1/'
    + 'LT05_L2SP_044029_20030827_20200904_02_T1_SR_B2.TIF',
):
    # this file is the canary in the coal mine
    # if you can't open this one you've got *issues* because it exists!
    # also the instantiation of the environment here
    # might help you turn on the switch of the credentials
    # but maybe that's just anecdotal i hate credential stuff SO MUCH
    # if anyone is reading this message i hope you're enjoying my typing
    # as i wait for my cluster to start up.... hmm....

    with rio.Env(aws_session):
        with rio.open(canary_file) as src:
            src.read(1)


def fix_link(url):
    '''
    Fix the url string to be appropriate for s3
    '''
    return url.replace('https://landsatlook.usgs.gov/data', 's3://usgs-landsat')


def cloud_qa(item):
    '''
    Given an item grab the cloud mask for that scene
    '''

    if isinstance(item, str):
        qa_path = item
    else:
        qa_path = fix_link(item._stac_obj.assets['SR_CLOUD_QA.TIF']['href'])
    cog_mask = xr.open_rasterio(qa_path).squeeze().drop('band')
    return cog_mask


def grab_ds(item, bands_of_interest, cog_mask, utm_zone, utm_letter):
    '''
    Package up a dataset of multiple bands from one scene, masked
    according to the QA/QC mask.

    Parameters
    ----------
    item : stac item
        The stac item for a specific scene
    bands_of_interest: list
        The list of bands you want to include in your dataset
        (e.g. ['SR_B1', 'SR_B2'])
    cog_mask: xarray data array
        The mask specific to this scene which you'll use to
        exclude poor quality observations
    utm_zone: str
        The UTM zone used in the projection of this landsat scene. This
        defined by USGS and sometimes doesn't match what you'd expect
        so is important to carry through in the processing.
    utm_letter: str
        The UTM zone used in the projection of this landsat scene. This
        defined by USGS and sometimes doesn't match what you'd expect
        so is important to carry through in the processing.

    Returns
    -------
    xarray dataset
        Dataset of dimensions x, y, bands (corresponding to bands_of_interest)
    '''

    if type(item) == str:
        url_list = [item + '_{}.TIF'.format(band) for band in bands_of_interest]
    else:
        url_list = [
            fix_link(item._stac_obj.assets['{}.TIF'.format(band)]['href'])
            for band in bands_of_interest
        ]
    da_list = []
    for url in url_list:
        da_list.append(rioxarray.open_rasterio(url, chunks={'x': 1024, 'y': 1024}))
    # combine into one dataset
    ds = xr.concat(da_list, dim='band').to_dataset(dim='band').rename({1: 'reflectance'})
    del da_list
    ds = ds.assign_coords({'band': bands_of_interest})
    ds = ds.where(ds != 0)
    ds = ds.where(cog_mask < 2)
    # scale the reflectance values according to recommendations from USGS
    # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products
    ds = ds.reflectance.to_dataset(dim='band') * 0.0000275 - 0.2
    ds.attrs["utm_zone_number"] = utm_zone
    ds.attrs["utm_zone_letter"] = utm_letter
    ds = calc_NDVI(ds)
    ds = calc_NDII(ds)
    gc.collect()
    return ds


def valid_pixel_mask(ds):
    return (~ds['SR_B1'].isnull()).astype(int).compute()


def average_stack_of_scenes(ds_list):
    '''
    Average across scenes. This will work the same regardless
    of whether your scenes are perfectly overlapping or they're offset.
    However, if they're offset it requires a merge and so the entire
    datacube (pre-collapsing) will be instantiated and might make
    your kernel explode.

    Parameters
    ----------
    ds_list : list
        List of xarray datsets to average.

    Returns
    -------
    xarray dataset
        Dataset of dimensions x, y, bands (corresponding to bands_of_interest)
    '''
    utm_zone = []
    utm_letter = []
    for ds in ds_list:
        utm_zone.append(ds.attrs['utm_zone_number'])
        utm_letter.append(ds.attrs['utm_zone_letter'])
    # WATCH OUT: you don't want to average scenes from multiple utm projections!!
    # thank goodness we have these two swanky assertions below
    # TODO: this could probably be moved to a test
    assert len(set(utm_zone)) == 1
    assert len(set(utm_letter)) == 1
    # less memory-intensive way of averaging
    full_ds = ds_list.pop().load()
    valid_pixel_count = valid_pixel_mask(full_ds)
    while ds_list:
        ds = ds_list.pop().load()
        full_ds += ds.compute()
        mask = valid_pixel_mask(ds).load()
        valid_pixel_count = (valid_pixel_count + mask).load()
        del mask
        del ds
    # divide by the number of active pixels to get your seasonal average
    full_ds = full_ds / valid_pixel_count
    del valid_pixel_count

    full_ds.attrs['utm_zone_number'] = utm_zone[0]
    full_ds.attrs['utm_zone_letter'] = utm_letter[0]
    return full_ds


def write_out(ds, mapper):
    '''
    Write out your final dataset.

    Parameters
    ----------
    ds : xarray dataset
        Final compiled xarray dataset
    mapper: mapper
        Mapper of location to write the dataset
    '''
    # encoding = {'reflectance': {'compressor': numcodecs.Blosc()}}
    ds.to_zarr(
        store=mapper,
        # encoding=encoding,
        mode='w',
    )


def access_credentials():
    '''
    Access the credentials you'll need for read/write permissions.
    This will access a file with your credentials in your home directory,
    so that file needs to exist in the correct format.
    *CAREFUL* This is brittle.

    Returns
    -------
    access_key_id : str
        Key ID for AWS credentials
    secret_access_key : str
        Secret access key for AWS credentials
    '''

    with open('/home/jovyan/.aws/credentials') as f:
        credentials = f.read().splitlines()
        access_key_id = credentials[1].split('=')[1].strip()
        secret_access_key = credentials[2].split('=')[1].strip()
    return access_key_id, secret_access_key


def grab_scene_coord_info(metadata):
    '''
    Grab latitude values for scene corners.
    '''

    lats = []
    for key in [
        'CORNER_LL_LAT_PRODUCT',
        'CORNER_LR_LAT_PRODUCT',
        'CORNER_UL_LAT_PRODUCT',
        'CORNER_UR_LAT_PRODUCT',
    ]:
        lats.append(float(metadata[key]))
    upper_right_corner_proj = (
        float(metadata['CORNER_UR_PROJECTION_X_PRODUCT']),
        float(metadata['CORNER_UR_PROJECTION_Y_PRODUCT']),
    )
    upper_right_corner_coords = (
        float(metadata['CORNER_UR_LON_PRODUCT']),
        float(metadata['CORNER_UR_LAT_PRODUCT']),
    )
    return lats, upper_right_corner_proj, upper_right_corner_coords


def get_scene_utm_info(url, json_client):

    '''
    Get the USGS-provided UTM zone and letter for the specific landsat scene

    Parameters
    ----------
    url: str
        root url to landsat scene

    Returns
    -------
    utm_zone: str
    utm_letter: str

    '''

    metadata_url = url + '_MTL.json'
    data = json_client.get_object(
        Bucket='usgs-landsat', Key=metadata_url[18:], RequestPayer='requester'
    )
    metadata = json.loads(data['Body'].read())
    utm_zone = metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['UTM_ZONE']

    lats, upper_right_corner_proj, upper_right_corner_coords = grab_scene_coord_info(
        metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']
    )

    if spans_utm_border(lats):
        utm_zone_letter = verify_projection(
            upper_right_corner_coords, upper_right_corner_proj, int(utm_zone)
        )
    else:
        (_x, _y, calculated_zone_number, utm_zone_letter) = utm.from_latlon(
            upper_right_corner_coords[1],
            upper_right_corner_coords[0],
            force_zone_number=int(utm_zone),
        )
    return utm_zone, utm_zone_letter


def calc_NDVI(ds):
    '''
    Calculate NDVI (Jordan 1969, Rouse et al 1974) based upon bands 3 and 4.
    *Note* only valid for landsat 5 and 7 right now.
    https://www.usgs.gov/core-science-systems/nli/landsat/landsat-normalized-difference-vegetation-index

    Parameters
    ----------
    ds: xarray Dataset
        dataset with six surface reflectance bands

    Returns
    -------
    ds: xarray Dataset
        dataset with NDVI added as variable
    '''

    nir = ds['SR_B4']
    red = ds['SR_B3']
    ds['NDVI'] = (nir - red) / (nir + red)
    return ds


def calc_NDII(ds):

    '''
    Calculate NDII (Hardisky et al, 1984) based upon bands 4 and 5. *Note* only valid
    for landsat 5 and 7 right now.


    Parameters
    ----------
    ds: xarray Dataset
        dataset with six surface reflectance bands

    Returns
    -------
    ds: xarray Dataset
        dataset with NDII added as variable
    '''
    nir = ds['SR_B4']
    swir = ds['SR_B5']
    ds['NDII'] = (nir - swir) / (nir + swir)

    return ds


def find_months_of_interest(row):
    '''
    Grab the growing season months based upon the
    landsat row.
    '''
    if row <= 40:
        # northern hemisphere
        return ['06', '07', '08']
    elif row >= 80:
        # southern hemisphere
        return ['12', '01', '02']
    else:
        # tropics
        return [f'{month:02}' for month in np.arange(1, 13)]


def make_datestamps(months, year):
    '''
    Construct the datestamp you'll search for in the landsat archive.
    We will grab landsat scenes corresponding to the year of interest,
    except for january/february which we will grab from the following year
    to ensure that the austral summer is assigned to the year of interest.
    i.e. calculation of biomass for 2003 in the southern hemisphere will
    include landsat scenes from jan/feb 2004.
    '''
    years = [year + 1 if int(month) < 3 else year for month in months]
    return ['{}{}'.format(year, month) for (year, month) in zip(years, months)]


def scene_seasonal_average(
    path,
    row,
    year,
    access_key_id,
    secret_access_key,
    aws_session,
    core_session,
    fs,
    write_bucket=None,
    bands_of_interest='all',
    landsat_generation='landsat-7',
):
    '''
    Given location/time specifications will grab all valid scenes,
    mask each according to its time-specific cloud QA and then
    return average across all masked scenes
    '''
    test_credentials(core_session)
    # all of this is just to get the right formatting stuff to access the scenes
    test_client = core_session.client('s3')
    if landsat_generation == 'landsat-7':
        landsat_bucket = 's3://usgs-landsat/collection02/level-2/standard/etm/{}/{:03d}/{:03d}/'
    elif landsat_generation == 'landsat-5':
        landsat_bucket = 's3://usgs-landsat/collection02/level-2/standard/tm/{}/{:03d}/{:03d}/'
    # find the right month keys based upon your landsat row
    months = find_months_of_interest(row)

    valid_files, ds_list = [], []

    if bands_of_interest == 'all':
        bands_of_interest = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    scene_stores = fs.ls(landsat_bucket.format(year, path, row))
    datestamps = make_datestamps(months, year)
    for scene_store in scene_stores:
        for datestamp in datestamps:
            if datestamp in scene_store:
                valid_files.append(scene_store)
    for file in valid_files:
        scene_id = file[-40:]
        url = 's3://{}/{}'.format(file, scene_id)
        utm_zone, utm_letter = get_scene_utm_info(url, test_client)
        cloud_mask_url = url + '_SR_CLOUD_QA.TIF'
        cog_mask = cloud_qa(cloud_mask_url)
        ds_list.append(grab_ds(url, bands_of_interest, cog_mask, utm_zone, utm_letter))
    seasonal_average = average_stack_of_scenes(ds_list)
    del ds_list
    if write_bucket is not None:
        # set where you'll save the final seasonal average
        url = f'{write_bucket}{path}/{row}/{year}/growing_season_reflectance.zarr'
        mapper = fs.get_mapper(url)
        write_out(seasonal_average.chunk({'x': 1024, 'y': 1024}), mapper)
        return seasonal_average.chunk({'x': 1024, 'y': 1024}).load()
    else:
        return seasonal_average.chunk({'x': 1024, 'y': 1024}).load()


scene_seasonal_average_delayed = dask.delayed(scene_seasonal_average)
