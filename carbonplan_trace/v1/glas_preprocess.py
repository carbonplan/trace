from datetime import datetime, timezone

import numpy as np
import xarray as xr
from scipy import optimize
from scipy.ndimage import gaussian_filter1d

import carbonplan_trace.v1.glas_height_metrics as ht
import carbonplan_trace.v1.utils as utils

SPEED_OF_LIGHT = 299792458  # m/s
SECONDS_IN_NANOSECONDS = 10 ** -9


def calculate_derived_variables(data, tiles):
    """
    Calculate derived variables in a xarray dataset containing glas data
    """
    # convert receiving waveform digital bins from 0-543 to corresponding distance from satellite
    data["rec_wf_sample_dist"] = (
        (data.rec_wf_sample_loc + data.rec_wf_response_end_time - data.tx_wf_peak_time)
        * SECONDS_IN_NANOSECONDS
        * SPEED_OF_LIGHT
    ) / 2

    # waveform extent here is calculated based on the signal beginning and end identified in the GLAH14 data
    data = ht.get_height_metric_value(ds=data, metric="wf_extent", recalc=True)
    data = ht.get_dist_metric_value(ds=data, metric="gaussian_fit_dist")

    data['glas_elev'] = data.elevation + data.elevation_correction

    srtm = utils.open_srtm_data(tiles=tiles)
    if srtm is not None:
        srtm_raw = utils.find_matching_records(data=srtm, lats=data.lat, lons=data.lon)
        srtm_elev = srtm_raw.srtm
    else:
        srtm_elev = np.nan

    data['srtm_elev'] = srtm_elev + data.delta_ellipse + data.geoid
    del srtm

    return data


def process_coordinates(ds):
    """
    Process lat/lon to get xy from Landsat images, process time from "seconds since 2000/1/1" to unix/epoch timestamp
    All inputs are xr dataarrays
    """
    # original time format is seconds elapsed since Jan 1 2000 12:00:00 UTC
    d0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    ds['time'] = ds.time + d0

    # get datetime object
    ds['datetime'] = xr.apply_ufunc(
        datetime.fromtimestamp, ds.time.fillna(d0), vectorize=True, dask='parallelized'
    )
    ds['datetime'] = ds.datetime.astype(np.datetime64)

    return ds


def get_mask(ds):
    """
    True in mask = records to use
    """
    # all non nulls in the GLAH14 dataset
    mask = ds.lat.notnull()
    mask.name = 'mask'

    # Harris et al 2021 filtering conditions listed in Supplementary Information
    mask = (
        mask
        & (ds.num_gaussian_peaks >= 2)  # have at least two peaks
        # max amplitude of waveform greater than 2x baseline noise
        & (ds.rec_wf.max(dim='rec_bin') >= (ds.noise_mean * 2))
        # discrepancy bt SRTM and GLAS derived elevation less than 30m
        & (abs(ds.srtm_elev.fillna(ds.glas_elev) - ds.glas_elev) <= 30)
        # signal beginning is less than 70m (otherwise indicates potential inference of signal)
        & (abs(ds.sig_begin_offset) <= 70)
        # signal end is less than 20 and greater than 1m (otherwise indicates sig end is improperly captured)
        & (abs(ds.sig_end_offset) <= 20)
        & (abs(ds.sig_end_offset) >= 1)
    )

    return mask.load()


def filter_large_leading_and_trailing_edge_extent(ds):
    metrics = [
        "leading_edge_extent",
        "trailing_edge_extent",
    ]
    ds = ht.get_all_height_metrics(
        ds=ds,
        metrics=metrics,
    )

    # waveform extent here is calculated based on the signal beginning and end identified in the GLAH14 data
    # and leading/trailing edge extents were calculated using the sum of up to 6 gaussian peaks (modeled waveform)
    mask = (
        # leading edge <= 50% of wf extent, otherwise indicates large differences in canopy height
        (ds.leading_edge_extent <= (ds.wf_extent * 0.5))
        # trailing edge <= 35% of wf extent, otherwise indicates impacts from high slope
        & (ds.trailing_edge_extent <= (ds.wf_extent * 0.35))
    )
    total = ds.dims['unique_index']
    ds = ds.where(mask, drop=True)
    remained = ds.dims['unique_index']
    print(
        f'after edge extent filtering, {remained} valid shots remained out of {total} ({100 - round(remained/total*100, 2)}%) filtered'
    )
    return ds


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-1 * ((x - mean) ** 2) / (2 * (stddev ** 2)))


def find_gaussian_fit_sigma(wf, default=3):
    """
    wf is a 1D array, not a vectorized function
    """
    x = np.arange(len(wf))
    y = wf - wf.min()  # optimizer can't deal with negative number
    try:
        popt, _ = optimize.curve_fit(gaussian, x, y, p0=[0.5, 25, default])
        sigma = popt[2]
    except RuntimeError:
        # print("Using default sigma")
        sigma = default

    return sigma


def smooth_wf(rec_wf, tx_wf, verbose=False):
    """
    Find sigma from transmitted waveform, and apply gaussian filter smoothing on the recieved waveform with said sigma
    """
    if np.any(np.isnan(rec_wf)):
        if verbose:
            print('skipping record in smooothing due to nans')
        return rec_wf

    # note: in Farina et al 2018 this value is directly taken from GLAH05 data Data_40HZ/Transmit_Energy/d_sigmaTr
    # re fitting here since the data would be in `ns` unit and requires some translation to be used directly here
    sigma = find_gaussian_fit_sigma(tx_wf)

    return gaussian_filter1d(input=rec_wf, sigma=sigma)


def select_valid_area(bins, wf, signal_begin_dist, signal_end_dist):
    """
    vectorized
    """
    # get mask of valid area
    valid = (bins > signal_begin_dist) & (bins < signal_end_dist)

    # set all invalid area to 0s and clip lower at 0
    wf = wf.where(valid, other=0).clip(min=0)

    return wf


def get_modeled_waveform(ds):
    """
    Sum up the gaussian peaks in GLAH14
    """
    # we have gaussian_fit_dist in meters, gaussian_amp in volts
    # gaussian sigma is in nanoseconds, switch to meters and divide by 2 to get 1 side stdev
    gaussian_sigma_in_m = (ds.gaussian_sigma * SECONDS_IN_NANOSECONDS * SPEED_OF_LIGHT) / 2
    modeled_wf = gaussian(
        x=ds.rec_wf_sample_dist,
        amplitude=ds.gaussian_amp,
        mean=ds.gaussian_fit_dist,
        stddev=gaussian_sigma_in_m,
    ).sum(dim='n_gaussian_peaks')

    # denoise
    modeled_wf = modeled_wf - ds.noise_mean

    # set the energy outside of signal begin/end to 0
    modeled_wf = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=modeled_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.sig_end_dist,
    )

    dims = ds.rec_wf.dims
    modeled_wf = modeled_wf.transpose(dims[0], dims[1])

    return modeled_wf


def get_smoothed_actual_waveform(ds):
    """
    Smooth and de-noise actual waveform from GLAH01
    """
    # apply gaussian filter to smooth
    processed_wf = xr.apply_ufunc(
        smooth_wf,
        ds.rec_wf,
        ds.tx_wf,
        input_core_dims=[["rec_bin"], ["tx_bin"]],
        output_core_dims=[["rec_bin"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": 1},
        output_dtypes=['float'],
    )

    # denoise
    processed_wf = processed_wf - ds.noise_mean

    # set the energy outside of signal begin/end to 0
    processed_wf = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.sig_end_dist,
    )

    dims = ds.rec_wf.dims
    processed_wf = processed_wf.transpose(dims[0], dims[1])

    return processed_wf


def preprocess(ds, min_lat, max_lat, min_lon, max_lon):
    # find a list of 10x10 degree tile names covering the bounding box
    # the ancillary data used in preprocess are stored as these 10x10 degree tiles
    tiles = utils.find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon)
    # calculate variables used in the rest of the preprocess
    ds = calculate_derived_variables(data=ds, tiles=tiles)

    # stack the record index and shot number together so we have a ~1D tabular data structure
    ds = ds.stack(unique_index=("record_index", "shot_number"))

    # apply filtering
    mask = get_mask(ds)
    total = ds.dims['unique_index']
    ds = ds.where(mask, drop=True)
    remained = ds.dims['unique_index']
    print(
        f'after initial filtering, {remained} valid shots remained out of {total} ({100 - round(remained/total*100, 2)}%) filtered'
    )

    # smooth and denoise waveform
    # print('before smoothing: current time is ', time.strftime("%H:%M:%S", time.localtime()))
    ds["processed_wf"] = get_smoothed_actual_waveform(ds)
    ds["modeled_wf"] = get_modeled_waveform(ds)
    # leading and trailing edge extent calculations require processed wf (which is time consuming to calculate)
    # thus, filtering is broken into two parts
    ds = filter_large_leading_and_trailing_edge_extent(ds)

    # preprocess the coordinate variables
    # print('before coordinates: current time is ', time.strftime("%H:%M:%S", time.localtime()))
    ds = process_coordinates(ds)

    return ds
