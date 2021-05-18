import boto3
from rasterio.session import AWSSession
from s3fs import S3FileSystem
from osgeo.gdal import VSICurlClearCache
import rasterio as rio
import xarray as xr
import dask
import os
import fsspec
import geopandas as gpd
import regionmask as rm
from matplotlib.pyplot import imshow
from intake import open_stac_item_collection
import numcodecs
import numpy as np
import rioxarray # for the extension to load
import matplotlib.pyplot as plt
import utm
import pandas as pd
from datetime import datetime
import json
import zarr
import awswrangler as wr


def test_credentials(aws_session, 
                            canary_file='s3://usgs-landsat/collection02/level-2/standard/'+\
                            'tm/2003/044/029/LT05_L2SP_044029_20030827_20200904_02_T1/'+\
                            'LT05_L2SP_044029_20030827_20200904_02_T1_SR_B2.TIF'):    
    # this file is the canary in the coal mine
    # if you can't open this one you've got *issues* because it exists!
    # also the instantiation of the environment here
    # might help you turn on the switch of the credentials
    # but maybe that's just anecdotal i hate credential stuff SO MUCH
    # if anyone is reading this message i hope you're enjoying my typing
    # as i wait for my cluster to start up.... hmm....

    with rio.Env(aws_session):
        with rio.open(canary_file) as src:
            profile = src.profile
            
            arr = src.read(1)


def fix_link(url):
    '''
    Fix the url string to be appropriate for s3
    '''
    return url.replace('https://landsatlook.usgs.gov/data', 's3://usgs-landsat')


def cloud_qa(item):
    '''
    Given an item grab the cloud mask for that scene
    '''

    if type(item)==str:
        qa_path = item
    else:
        qa_path = fix_link(item._stac_obj.assets['SR_CLOUD_QA.TIF']['href'])
    cog_mask = xr.open_rasterio(qa_path).squeeze().drop('band')
    return cog_mask


def grab_ds(item, bands_of_interest, cog_mask, utm_zone):
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
    utm_zone: int
        The UTM zone used in the projection of this landsat scene. This
        defined by USGS and sometimes doesn't match what you'd expect
        so is important to carry through in the processing.
    
    Returns
    -------
    xarray dataset
        Dataset of dimensions x, y, bands (corresponding to bands_of_interest)
    '''

    if type(item) == str:
        url_list = [item+'_{}.TIF'.format(band) for band in bands_of_interest]
    else:
        url_list = [fix_link(item._stac_obj.assets['{}.TIF'.format(band)]['href']) for band in bands_of_interest]
    da_list = []
    for url in url_list:
        da_list.append(rioxarray.open_rasterio(url, chunks={'x': 1024,
                                                    'y': 1024}))

    # combine into one dataset
    ds = xr.concat(da_list, dim='band').to_dataset(dim='band').rename({1: 'reflectance'})
    ds = ds.assign_coords({'band': bands_of_interest})
    # fill value is 0; let's switch it to nan
    ds = ds.where(ds != 0)  
    ds = ds.where(cog_mask<2)
    ds.attrs["utm_zone"] = utm_zone
    ds = calc_NDVI(ds)
    ds = calc_NDII(ds)
    return ds


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
    for ds in ds_list:
        utm_zone.append(ds.attrs['utm_zone'])
    if len(set(utm_zone))>1:
        print('WATCH OUT: youre averaging scenes from multiple utm projections!!')
        
    full_ds = xr.concat(ds_list, dim='scene').mean(dim='scene')
    full_ds.attrs['utm_zone'] = utm_zone[0]
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
    encoding = {'reflectance': {'compressor': numcodecs.Blosc()}}
    ds.to_zarr(store=mapper,
                        encoding=encoding, 
                         mode='w')

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
        access_key_id = credentials[1].split('=')[1]
        secret_access_key = credentials[2].split('=')[1]
    return access_key_id, secret_access_key


def get_scene_utm_zone(url):

    '''
    Get the USGS-provided UTM zone for the specific landsat scene

    Parameters
    ----------
    url: str
        root url to landsat scene

    Returns
    -------
    utm_zone: str

    '''

    metadata_url = url+'_MTL.json'
    json_client = boto3.client('s3')
    data = json_client.get_object(Bucket='usgs-landsat',
                                 Key=metadata_url[18:],
                                 RequestPayer='requester')
    metadata = json.loads(data['Body'].read())
    utm_zone = metadata['LANDSAT_METADATA_FILE']['PROJECTION_ATTRIBUTES']['UTM_ZONE']
    return utm_zone

    
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

    nir = ds.sel(band='SR_B4')['reflectance'].drop('band')
    red = ds.sel(band='SR_B3')['reflectance'].drop('band')
    ds['NDVI'] = ( nir - red ) / ( nir + red)
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
    nir = ds.sel(band='SR_B4')['reflectance'].drop('band')
    swir = ds.sel(band='SR_B5')['reflectance'].drop('band')
    ds['NDII'] = ( nir - swir ) / ( nir + swir)
    return ds

@dask.delayed
def scene_seasonal_average(path, row, year, bucket, access_key_id, secret_access_key,
                           bands_of_interest='all', season='JJA'):
    '''
    Given location/time specifications will grab all valid scenes,
    mask each according to its time-specific cloud QA and then 
    return average across all masked scenes
    '''
    aws_session = AWSSession(boto3.Session(aws_access_key_id=access_key_id,
                                          aws_secret_access_key=secret_access_key),
                            requester_pays=True)
    fs = S3FileSystem(key=access_key_id,
                    secret=secret_access_key, requester_pays=True)

    with dask.config.set(scheduler='single-threaded'): # this? **** #threads #single-threaded # threads??
        with rio.Env(aws_session):
            test_credentials(aws_session)

            # set where you'll save the final seasonal average
            url = f'{bucket}{path}/{row}/{year}/{season}_reflectance.zarr'
            mapper = fs.get_mapper(url) #used to be fsspec
            # all of this is just to get the right formatting stuff to access the scenes

            landsat_bucket = 's3://usgs-landsat/collection02/level-2/standard/tm/{}/{:03d}/{:03d}/'
            month_keys = {'JJA': ['06', '07', '08']}
            valid_files, ds_list = [], []

            if bands_of_interest=='all':
                bands_of_interest = ['SR_B1', 'SR_B2', 'SR_B3', 
                                         'SR_B4', 'SR_B5', 'SR_B7']
            scene_stores = fs.ls(landsat_bucket.format(year, path, row))
            summer_datestamps = ['{}{}'.format(year, month) for month in month_keys[season]]
            for scene_store in scene_stores:
                for summer_datestamp in summer_datestamps:
                    if summer_datestamp in scene_store:
                        valid_files.append(scene_store)
            for file in valid_files:
                scene_id = file[-40:]
                url = 's3://{}/{}'.format(file, scene_id)
                utm_zone = get_scene_utm_zone(url)
                cloud_mask_url = url+'_SR_CLOUD_QA.TIF'
                cog_mask = cloud_qa(cloud_mask_url)
                ds_list.append(grab_ds(url, bands_of_interest, cog_mask, utm_zone))
            seasonal_average = average_stack_of_scenes(ds_list)
            
            write_out(seasonal_average.chunk({'band': 6, 'x': 1024, 'y': 1024}), mapper)
            return url