import numpy as np
import xarray as xr
from scipy import optimize
from scipy.ndimage import gaussian_filter1d

SPEED_OF_LIGHT = 299792458  # m/s
SECONDS_IN_NANOSECONDS = 10 ** -9


def calculate_derived_variables(ds):
    """
    Calculate derived variables in a xarray dataset containing glas data
    """
    # convert receiving waveform digital bins from 0-543 to corresponding distance from satellite
    ds["rec_wf_sample_distance"] = (
        (ds.rec_wf_sample_loc + ds.rec_wf_response_end_time - ds.tx_wf_peak_time)
        * SECONDS_IN_NANOSECONDS
        * SPEED_OF_LIGHT
    ) / 2

    # calculate the bias between reference range to the bottom of received wf
    ds["ref_range_bias"] = ds.rec_wf_sample_distance.max(dim="rec_bin") - ds.ref_range

    # convert offsets to distance from satellite
    ds["sig_begin_distance"] = ds.sig_begin_offset + ds.ref_range + ds.ref_range_bias
    ds["sig_end_distance"] = ds.sig_end_offset + ds.ref_range + ds.ref_range_bias
    ds["centroid_distance"] = ds.centroid_offset + ds.ref_range + ds.ref_range_bias

    return ds


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-1 * ((x - mean) ** 2) / (2 * (stddev ** 2)))


def find_gaussian_fit_sigma(wf, default=3):
    """
    wf is a 1D array, not a vectorized function
    """
    x = np.arange(len(wf))
    y = wf - wf.min()  # optimizezr can't deal with negative number
    try:
        popt, _ = optimize.curve_fit(gaussian, x, y, p0=[0.5, 25, 1])
        sigma = popt[2]
    except RuntimeError:
        print("Using default sigma")
        sigma = default

    return sigma


def smooth_wf(rec_wf, tx_wf):
    """
    Find sigma from transmitted waveform, and apply gaussian filter smoothing on the recieved waveform with said sigma
    """
    sigma = find_gaussian_fit_sigma(tx_wf)

    return gaussian_filter1d(input=rec_wf, sigma=sigma)


def select_valid_area(bins, wf, signal_begin_distance, signal_end_distance):
    """
    vectorized
    """
    # get mask of valid area
    valid = (bins > signal_begin_distance) & (bins < signal_end_distance)

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
        output_dtypes=np.float64,
    )

    # denoise
    processed_wf = processed_wf - ds.noise_mean

    # set the energy outside of signal begin/end to 0
    processed_wf = select_valid_area(
        bins=ds.rec_wf_sample_distance,
        wf=processed_wf,
        signal_begin_distance=ds.sig_begin_distance,
        signal_end_distance=ds.sig_end_distance,
    )

    dims = ds.rec_wf.dims
    processed_wf = processed_wf.transpose(dims[0], dims[1], dims[2])

    return processed_wf


def preprocess(ds):
    ds = calculate_derived_variables(ds)
    ds["processed_wf"] = preprocess_wf(ds)

    return ds
