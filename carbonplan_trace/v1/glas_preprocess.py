from datetime import datetime, timezone

import numpy as np
import xarray as xr
from scipy import optimize
from scipy.ndimage import gaussian_filter1d

import carbonplan_trace.v1.glas_height_metrics as ht
from carbonplan_trace.v1.utils import get_transformer, get_x_from_latlon, get_y_from_latlon

SPEED_OF_LIGHT = 299792458  # m/s
SECONDS_IN_NANOSECONDS = 10 ** -9


def calculate_derived_variables(ds):
    """
    Calculate derived variables in a xarray dataset containing glas data
    """
    # convert receiving waveform digital bins from 0-543 to corresponding distance from satellite
    ds["rec_wf_sample_dist"] = (
        (ds.rec_wf_sample_loc + ds.rec_wf_response_end_time - ds.tx_wf_peak_time)
        * SECONDS_IN_NANOSECONDS
        * SPEED_OF_LIGHT
    ) / 2

    ds["gaussian_fit_dist"] = ht.get_gaussian_fit_dist(ds)
    ds["sig_begin_dist"] = ht.get_sig_begin_dist(ds)
    ds["sig_end_dist"] = ht.get_sig_end_dist(ds)
    ds["ground_peak_dist"] = ht.get_ground_peak_dist(ds)

    ds["wf_extent"] = ht.sig_beg_to_sig_end_ht(ds)
    ds["leading_edge_extent"] = ht.get_leading_edge_extent(ds)
    ds["trailing_edge_extent"] = ht.get_trailing_edge_extent(ds)

    return ds


def process_coordinates(ds):
    """
    Process lat/lon to get xy from Landsat images, process time from "seconds since 2000/1/1" to unix/epoch timestamp
    All inputs are xr dataarrays
    """
    ds['x'] = xr.apply_ufunc(
        get_x_from_latlon,
        ds.lat,
        ds.lon,
        get_transformer(),
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': 1},
        output_dtypes=np.float64,
    )

    ds['y'] = xr.apply_ufunc(
        get_y_from_latlon,
        ds.lat,
        ds.lon,
        get_transformer(),
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': 1},
        output_dtypes=np.float64,
    )

    # original time format is seconds elapsed since Jan 1 2000 12:00:00 UTC
    d0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    ds['time'] = ds.time + d0

    return ds


def get_mask(ds):
    """
    True in mask = records to use
    """

    def get_pct_false(m):
        return round(100.0 - m.sum().values / m.count().values * 100, 2)

    # all non nulls in the GLAH14 dataset
    mask = ~ds.lat.isnull()
    mask.name = 'mask'
    m1 = get_pct_false(mask)
    print(f'filtering out {m1}% of records due to null GLAH14 data')

    # Harris et al 2021 filtering conditions listed in Supplementary Information
    mask = mask & (ds.num_gaussian_peaks >= 2)  # have at least two peaks
    m2 = get_pct_false(mask)
    print(f'filtering out {m2-m1}% of records due to number of gaussian peaks')
    m1 = m2

    mask = mask & (
        ds.rec_wf.max(dim='rec_bin') >= (ds.noise_mean * 2)
    )  # max amplitude of waveform greater than 2x baseline noise
    m2 = get_pct_false(mask)
    print(f'filtering out {m2-m1}% of records due to max amplitude too small')
    m1 = m2

    # mask = mask & (
    #     abs(ds.elevation_SRTM - (ds.elevation + ds.elevation_correction)) <= 30
    # )  # discrepancy bt SRTM and GLAS derived elevation less than 30m
    # m2 = get_pct_false(mask)
    # print(f'filtering out {m2-m1}% of records due to discrepancy in SRTM and GLAS elev')
    # m1 = m2

    mask = (
        mask
        # signal beginning is less than 70m (otherwise indicates potential inference of signal)
        & (abs(ds.ground_peak_dist - ds.sig_begin_dist) <= 70)
        # signal end is less than 20 and greater than 1m (otherwise indicates sig end is improperly captured)
        & (abs(ds.ground_peak_dist - ds.sig_end_dist) <= 20)
        & (abs(ds.ground_peak_dist - ds.sig_end_dist) >= 1)
    )
    m2 = get_pct_false(mask)
    print(f'filtering out {m2-m1}% of records due to signal beginning or end out of bounds')
    m1 = m2

    mask = mask & (
        ds.leading_edge_extent <= (ds.wf_extent * 0.5)
    )  # leading edge <= 50% of wf extent, otherwise indicates large distances in canopy height
    m2 = get_pct_false(mask)
    print(f'filtering out {m2-m1}% of records due to leading edge extent too large')
    m1 = m2

    mask = mask & (
        ds.trailing_edge_extent <= (ds.wf_extent * 0.35)
    )  # trailing edge <= 35% of wf extent, otherwise indicates impacts from high slope
    m2 = get_pct_false(mask)
    print(f'filtering out {m2-m1}% of records due to trailing edge extent too large')

    return mask


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-1 * ((x - mean) ** 2) / (2 * (stddev ** 2)))


def find_gaussian_fit_sigma(wf, default=3):
    """
    wf is a 1D array, not a vectorized function
    """
    x = np.arange(len(wf))
    y = wf - wf.min()  # optimizezr can't deal with negative number
    try:
        popt, _ = optimize.curve_fit(gaussian, x, y, p0=[0.5, 25, default])
        sigma = popt[2]
    except RuntimeError:
        print("Using default sigma")
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

    # TODO: in Farina et al 2018 this value is directly taken from GLAH05 data Data_40HZ/Transmit_Energy/d_sigmaTr
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


def preprocess_wf(ds):
    """
    Smooth and de-noise received waveform, input is an xarray dataset with rec_wf, tx_wf, and noise_mean as dataarrays.
    Output is a dataarray containing the processed received waveform
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
        output_dtypes=float,
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


def preprocess(ds):
    # variables used in the rest of the preprocess
    ds = calculate_derived_variables(ds)

    # stack the record index and shot number together so we have a ~1D tabular data structure
    ds = ds.stack(unique_index=("record_index", "shot_number"))

    # apply filtering
    ds["mask"] = get_mask(ds)
    total = ds.noise_mean.fillna(0).count().values
    print('applying mask')
    ds = ds.where(ds.mask, drop=True)
    remained = ds.noise_mean.fillna(0).count().values
    print(
        f'after filtering, {remained} valid shots remained out of {total} ({100 - round(remained/total*100, 2)}%) filtered'
    )

    # smooth and denoise waveform
    ds["processed_wf"] = preprocess_wf(ds)

    # preprocess the coordinate variables
    ds = process_coordinates(ds)

    return ds
