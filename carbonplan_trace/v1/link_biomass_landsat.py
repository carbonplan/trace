import json
import os
from datetime import datetime

import awswrangler as wr
import boto3
import dask
import fsspec
import geopandas as gpd
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import rasterio as rio
import regionmask as rm
import rioxarray  # for the extension to load
import utm
import xarray as xr
import zarr
from intake import open_stac_item_collection
from matplotlib.pyplot import imshow
from osgeo.gdal import VSICurlClearCache
from rasterio.session import AWSSession
from s3fs import S3FileSystem


def add_landsat_utm_zone(scene_gdf):
    '''
    Grab sample file for each landsat scene and
    extract the utm zone then add that to the
    scene geodataframe

    Parameters
    ----------
    scene_gdf : geopandas geodataframe
        geodataframe whose rows are entries for each row/path scene

    Returns
    -------
    scene_gdf : geopandas geodataframe
        same geodataframe but with new column including the USGS-provided,
        scene-specific UTM zone

    '''
    scene_gdf['landsat_utm_zone'] = ""
    for scene_id in scene_gdf.index:
        scene_row = washington_scenes.loc[scene_id]['ROW']
        scene_path = washington_scenes.loc[scene_id]['PATH']
        url = 's3://carbonplan-climatetrace/v1/{}/{}/2004/JJA_reflectance.zarr'.format(
            scene_path, scene_row
        )
        landsat_utm_zone = xr.open_zarr(fs.get_mapper(url)).utm_zone
        washington_scenes.loc[scene_id, 'landsat_utm_zone'] = int(landsat_utm_zone)
    return scene_gdf


def convert_to_utm(df):
    '''
    Given dataframe with lat/lon coordinates of GLAS shots, project
    into the correct x/y coordinates to link to the Landsat scene.

    Parameters
    ----------
    df : pandas dataframe
        geodataframe whose rows are entries for each row/path scene. Must
        include variables 'lon', 'lat', and a utm zone called 'landsat_utm_zone'

    Returns
    -------
    df : pandas dataframe
        The projected information for each shot
    '''

    return utm.from_latlon(df['lat'], df['lon'], force_zone_number=df['landsat_utm_zone'])


def grab_year(df):
    '''
    Access calendar year in timestamp within a dataframe
    '''
    return datetime.fromtimestamp(df['time']).year


def add_projection_info(df):
    '''
    Given dataframe with lat/lon coordinates of GLAS shots,
    add UTM zone (as specified by USGS landsat product) and use
    that zone to translate GLAS shot location into projected x/y space

    Parameters
    ----------
    df : pandas dataframe
        geodataframe whose rows are entries for each row/path scene. Must
        include variables 'lon', 'lat', and a utm zone called 'landsat_utm_zone'

    Returns
    -------
    updated_df : pandas dataframe
        The df with the projected location of each GLAS shot
    '''

    projection_info = df.apply(convert_to_utm, axis=1).to_list()
    projected_column_names = ['proj_x', 'proj_y', 'utm_zone', 'utm_letter']
    projection_df = pd.DataFrame(projection_info, columns=projected_column_names, index=df.index)
    updated_df = pd.concat([df, projection_df], axis=1)
    return updated_df


def build_url(df):
    '''
    Add landsat url to dataframe
    Parameters
    ----------
    df : pandas dataframe
        geodataframe whose rows are entries for each row/path scene. Must
        include variables 'PATH', 'ROW', and 'year'

    Returns
    -------
    df : pandas dataframe
        The df with a the url for the landsat scene encompassing each shot
    '''
    return 's3://carbonplan-climatetrace/v1/{}/{}/{}/JJA_reflectance.zarr'.format(
        df['PATH'], df['ROW'], df['year']
    )


def add_linking_info(df):
    '''
    Given dataframe with lat/lon coordinates of GLAS shots,
    add all necessary linking info- projection, year, and url
    (which will act as the way to add landsat)
    Parameters
    ----------
    df : pandas dataframe
        geodataframe whose rows are entries for each row/path scene. Must
        include variables 'lon', 'lat', and a utm zone called 'landsat_utm_zone'

    Returns
    -------
    df : pandas dataframe
        The df with the projected location of each GLAS shot
    '''
    df = add_projection_info(df)
    df['year'] = df.apply(grab_year, axis=1)
    df['url'] = df.apply(build_url, axis=1)
    return df


def create_combined_landsat_biomass_df(
    landsat_ds, biomass_df, biomass_variables=['biomass', 'glas_elev', 'ecoregion']
):
    '''
    Add landsat info for each biomass entry

    Parameters
    ----------
    landsat_ds : xarray dataset
        landsat dataset to extract information from
    biomass_df : pandas dataframe
        dataframe with biomass and all information for linking to
        landsat
    biomass_variables : list
        List of variables to retain from original biomass_df
    Returns
    -------
    out_df : pandas dataframe
        The df with the biomass (and potentially other variables) and
        the corresponding landsat data
    '''
    selected_landsat = (
        landsat_ds.sel(
            x=xr.DataArray(biomass_df['proj_x'].values, dims='shot'),
            y=xr.DataArray(biomass_df['proj_y'].values, dims='shot'),
            method='nearest',
        )
        .to_dataframe()
        .drop(['x', 'y'], axis=1)
    )
    selected_landsat.index = biomass_df.index
    out_df = pd.concat([biomass_df[biomass_variables], selected_landsat], axis=1)
    return out_df
