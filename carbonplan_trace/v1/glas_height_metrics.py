from datetime import datetime, timezone
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def sig_beg_to_sig_end_ht(ds):
    """
    Waveform extent from signal beginning to end
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='sig_end_dist'
    ).clip(min=0)


def sig_beg_to_ground_ht(ds):
    """
    Top of tree height (meters). That is, distance from GLAS signal beginning to last peak (assumed to be ground peak).
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='ground_peak_dist'
    )


def sig_beg_to_adj_ground_ht(ds):
    """
    Height in meters from GLAS signal beginning to whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='adj_ground_peak_dist'
    )


def quadratic_mean_to_adj_ground_peak_actual_wf_ht(ds):
    """
    Quadratic mean distance of the waveform from ground peak to signal beginning (meters).
    Ground peak defined as whichever of the two lowest peaks has greater amplitude using actual waveform.
    """
    return get_heights_from_distance(
        ds, top_metric='quadratic_mean_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def mean_to_adj_ground_peak_actual_wf_ht(ds):
    """
    Mean distance of the waveform from ground peak to signal beginning (meters).
    Ground peak defined as whichever of the two lowest peaks has greater amplitude using actual waveform.
    """
    return get_heights_from_distance(
        ds, top_metric='mean_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def centroid_to_adj_ground_ht(ds):
    """
    Height of median energy (meters). The distance, in meters, between the height of the waveform centroid
    and the height of the ground peak. Ground peak defined as whichever of the two lowest peaks has greater
    amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='centroid_dist', bottom_metric='adj_ground_peak_dist'
    )


def ratio_centroid_to_max_ht(ds):
    """
    Ratio of (HOME)_Yavaşlı to maximum vegetation height. Maximum vegetation height calculated as the
    distance from signal beginning to the ground peak, with the ground peak defined as whichever of the two
    lowest peaks has greater amplitude.
    """
    centroid_ht = centroid_to_adj_ground_ht(ds)
    max_ht = sig_beg_to_adj_ground_ht(ds)

    return centroid_ht / max_ht


def adj_ground_to_sig_end_ht(ds):
    """
    Distance, in meters, from the ground peak to the signal end. Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='adj_ground_peak_dist', bottom_metric='sig_end_dist'
    )


def sig_beg_to_highest_energy_ht(ds):
    """
    Distance between the height of signal beginning and the height of wf_max_e
    """
    return get_heights_from_distance(ds, top_metric='sig_begin_dist', bottom_metric='wf_max_e_dist')


def start_to_centroid_adj_ground_ht(ds):
    """
    Vertical distance, in meters, between the height of the ground peak and the height of the ground signal start.
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.

    Based on Lefsky et al 1999 "Surface Lidar Remote Sensing of Basal Area and Biomass in Deciduous Forests of Eastern Maryland, USA"
    The start of ground peak is defined by "the posterior half of the ground return is copied and flipped vertically to define the anterior
    half of the ground return" Figure 2a. Thus, this distance is equal to height of ground peak to signal end
    """
    return adj_ground_to_sig_end_ht(ds)


def sig_beg_to_start_of_ground_peak_ht(ds):
    """
    The distance from the signal beginning to the start of the ground peak (meters). Ground peak assumed to be the last peak.
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='start_of_ground_peak_dist'
    )


# TODO: change the next set of functions to take in pct as a param and avoid repeating
def pct_25_to_adj_ground_peak_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_25_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def pct_50_to_adj_ground_peak_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_50_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def pct_75_to_adj_ground_peak_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_75_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def pct_90_to_adj_ground_peak_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_90_dist', bottom_metric='adj_ground_peak_dist_actual_wf'
    )


def pct_05_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_05_dist', bottom_metric='sig_end_dist')


def pct_10_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_10_dist', bottom_metric='sig_end_dist')


def pct_20_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_20_dist', bottom_metric='sig_end_dist')


def pct_25_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_25_dist', bottom_metric='sig_end_dist')


def pct_30_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_30_dist', bottom_metric='sig_end_dist')


def pct_50_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_50_dist', bottom_metric='sig_end_dist')


def pct_75_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_75_dist', bottom_metric='sig_end_dist')


def pct_80_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_80_dist', bottom_metric='sig_end_dist')


def pct_90_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_90_dist', bottom_metric='sig_end_dist')


def pct_10_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_90_dist_modeled_wf', bottom_metric='sig_end_dist'
    )


def pct_25_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_75_dist_modeled_wf', bottom_metric='sig_end_dist'
    )


def pct_50_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_50_dist_modeled_wf', bottom_metric='sig_end_dist'
    )


def pct_60_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_40_dist_modeled_wf', bottom_metric='sig_end_dist'
    )


def pct_10_of_sig_beg_and_adj_ground_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds,
        top_metric='pct_10_of_sig_beg_and_adj_ground_dist_actual_wf',
        bottom_metric='adj_ground_peak_dist_actual_wf',
    )


def pct_80_of_sig_beg_and_adj_ground_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds,
        top_metric='pct_80_of_sig_beg_and_adj_ground_dist_actual_wf',
        bottom_metric='adj_ground_peak_dist_actual_wf',
    )


def pct_90_of_sig_beg_and_adj_ground_actual_wf_ht(ds):
    return get_heights_from_distance(
        ds,
        top_metric='pct_90_of_sig_beg_and_adj_ground_dist_actual_wf',
        bottom_metric='adj_ground_peak_dist_actual_wf',
    )


def pct_05_canopy_ht(ds):
    """
    the distance, in meters, between 0.5 m above the ground peak and the height at which 10% of
    the waveform energy from 0.5 m above the ground peak to signal beginning has been reached.
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='pct_05_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_10_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_10_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_20_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_20_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_30_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_30_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_50_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_50_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_75_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_75_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def pct_80_canopy_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_80_canopy_dist', bottom_metric='bottom_of_canopy_dist'
    )


def get_leading_edge_extent(ds):
    """
    the difference between the signal beginning (closest point to satellite that crossed noise threshold) and the H90
    (the height at which 90% of energy was reached from signal end to signal beginning, H90 is closer to satellite compared
    to H10). using modeled waveform here since the original Baccini processing used modeled waveform.
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='pct_90_dist_modeled_wf'
    )


def get_trailing_edge_extent(ds):
    """
    the difference between the signal end (furtherst point to satellite that crossed noise threshold) and the H10
    (the height at which 10% of energy was reached from signal end to signal beginning, H90 is closer to satellite compared
    to H10). using modeled waveform here since the original Baccini processing used modeled waveform.
    """
    return get_heights_from_distance(
        ds, top_metric='pct_10_dist_modeled_wf', bottom_metric='sig_end_dist'
    )


def get_sig_begin_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    return ds.sig_begin_offset + ds.rec_wf_sample_dist.max(dim="rec_bin")


def get_sig_end_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    return ds.sig_end_offset + ds.rec_wf_sample_dist.max(dim="rec_bin")


def get_centroid_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    return ds.centroid_offset + ds.rec_wf_sample_dist.max(dim="rec_bin")


def get_gaussian_fit_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    return ds.gaussian_mu + ds.rec_wf_sample_dist.max(dim="rec_bin")


def get_percentile_dist(ds, percentile):
    """
    the distance at which x% of the total waveform energy from signal beginning to signal end has been reached.
    energy accumulated relative to signal end.
    as an example, 25th percentile dist should be larger than 50th percentile dist, where larger dist = further
    to the satellite = lower elevation on earth
    percentiles is a list in hundredth format (e.g. to get 10th percentile input value 10)
    """
    cumsum = ds['processed_wf'].cumsum(dim="rec_bin")
    target = ds['processed_wf'].sum(dim="rec_bin") * percentile / 100.0

    return ds.rec_wf_sample_dist.where(cumsum > target).max(dim="rec_bin")


def get_percentile_dist_modeled_wf(ds, percentile):
    """
    the distance at which x% of the total waveform energy from signal beginning to signal end has been reached.
    energy accumulated relative to signal end.
    as an example, 25th percentile dist should be larger than 50th percentile dist, where larger dist = further
    to the satellite = lower elevation on earth
    percentiles is a list in hundredth format (e.g. to get 10th percentile input value 10)
    """
    cumsum = ds['modeled_wf'].cumsum(dim="rec_bin")
    target = ds['modeled_wf'].sum(dim="rec_bin") * percentile / 100.0

    return ds.rec_wf_sample_dist.where(cumsum > target).max(dim="rec_bin")


def get_percentile_of_sig_beg_and_adj_ground_dist_actual_wf(ds, percentile):
    """
    the distance, in meters, between the ground peak and the height at which 10% of the waveform energy
    from ground peak to signal beginning has been reached. Waveform energy accumulation relative to
    ground peak. Ground peak defined as whichever of the two lowest peaks has greater amplitude using actual waveform.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist_actual_wf,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    total = sig_beg_to_adj_ground.sum(dim="rec_bin")
    cumsum = sig_beg_to_adj_ground.cumsum(dim="rec_bin")
    target = total * percentile / 100.0

    return ds.rec_wf_sample_dist.where(cumsum > target).max(dim="rec_bin")


def get_percentile_canopy_dist(ds, percentile):
    """
    the height at which 10% of the waveform energy from 0.5 m above the ground peak
    to signal beginning has been reached
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='bottom_of_canopy_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    canopy = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.bottom_of_canopy_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    canopy = canopy.transpose(dims[0], dims[1])

    total = canopy.sum(dim="rec_bin")
    cumsum = canopy.cumsum(dim="rec_bin")
    target = total * percentile / 100.0

    return ds.rec_wf_sample_dist.where(cumsum > target).max(dim="rec_bin")


def get_ground_peak_dist(ds):
    """
    the lowest of gaussian fit peaks from GLAH14
    """
    # lowest elevation = largest distance
    return ds.gaussian_fit_dist.max(dim="n_gaussian_peaks")


def get_first_peak_dist(ds):
    """
    the peak closest to satellite from GLAH14
    """
    # closet to satellite = smallest distance
    return ds.gaussian_fit_dist.min(dim="n_gaussian_peaks")


def get_start_of_ground_peak_dist(ds):
    """
    Start of ground peak defined after Lefsky et al 1999 "Surface Lidar Remote Sensing of Basal Area and Biomass in Deciduous Forests of Eastern Maryland, USA"
    "the posterior half of the ground return is copied and flipped vertically to define the anterior
    half of the ground return" Figure 2a
    """
    ground_peak_trailing_extent = get_heights_from_distance(
        ds, top_metric='ground_peak_dist', bottom_metric='sig_end_dist'
    )

    return ds.ground_peak_dist - ground_peak_trailing_extent


def get_adj_ground_peak_dist(ds):
    """
    the centroid position of whichever of the two lowest fitted Gaussian peaks has greater amplitude, as defined by Rosette, North, and Suarez (2008)
    """
    # find the larger peak between the bottom two
    # We have a filter where we only process records with at least 2 peaks -- fillna is needed here because argmax doesn't deal with all nans
    loc = (
        ds.gaussian_amp.isel(n_gaussian_peaks=slice(2))
        .fillna(0)
        .argmax(dim="n_gaussian_peaks")
        .compute()
    )
    return ds.gaussian_fit_dist.isel(n_gaussian_peaks=loc)


def get_adj_ground_peak_dist_actual_wf(ds):
    """
    the position of whichever of the two lowest peaks has greater amplitude. peaks are identified from the smoothed, actual waveform
    """
    # TODO: allow some buffer between the signal end and the ground peak as described in Sun et al 2008
    # (Forest vertical structure from GLAS: An evaluation using LVIS and SRTM data)
    # the buffer in Sun et al is the "half width of the transmitted laser pulse"
    wf = ds.processed_wf

    diff_previous = wf - wf.shift(rec_bin=1)
    diff_next = wf - wf.shift(rec_bin=-1)
    peak_mask = ((diff_previous > 0) & (diff_next > 0)).compute()

    lowest_peak = peak_mask.argmax(dim='rec_bin')
    peak_mask[lowest_peak] = False
    second_lowest_peak = peak_mask.argmax(dim='rec_bin')
    lowest_amp = wf.isel(rec_bin=lowest_peak)
    second_lowest_amp = wf.isel(rec_bin=second_lowest_peak)

    result = xr.where(
        lowest_amp >= second_lowest_amp, x=lowest_peak, y=second_lowest_peak
    ).compute()

    return ds.rec_wf_sample_dist.isel(rec_bin=result)


def get_bottom_of_canopy_dist(ds):
    """
    0.5 m above the ground peak. Ground peak defined as whichever of the two lowest peaks has greater amplitude using actual wf
    """
    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')

    return ds.adj_ground_peak_dist_actual_wf - 0.5


def get_quadratic_mean_dist(ds):
    """
    From Neigh et al 2013: quadratic mean height of the waveform, calculated as the
    square root [∑ (normalized amplitude in a given canopy height bin) × (height of bin)**2]
    Original citation from Lefsky et al 1999. From signal beginning to adj ground peak identified using actual wf.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist_actual_wf,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    bins = ds.rec_wf_sample_dist

    # equivalent to
    # numerator = (np.square(bins) * sig_beg_to_adj_ground).sum(dim='rec_bin')
    # denom = sig_beg_to_adj_ground.sum(dim='rec_bin')
    # np.sqrt(numerator / denom)
    return np.sqrt(np.square(bins).weighted(sig_beg_to_adj_ground).mean("rec_bin"))


def get_mean_dist(ds):
    """
    Mean height of the waveform from the signal beginning to adj ground peak identified using actual wf
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist_actual_wf,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    bins = ds.rec_wf_sample_dist

    return bins.weighted(sig_beg_to_adj_ground).mean("rec_bin")


def get_wf_max_e_dist(ds):
    """
    distance at which max waveform energy occurs
    """
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_dist

    return bins.isel(rec_bin=wf.argmax(dim="rec_bin").compute())


def get_leading_edge_dist(ds):
    """
    the first elevation at which the waveform is half of the maximum signal above the background noise value
    highest elevation = smallest distance from satellite
    """
    energy_threshold = (ds.rec_wf.max(dim="rec_bin") - ds.noise_mean) / 2.0
    return ds.rec_wf_sample_dist.where(ds.rec_wf >= energy_threshold).min(dim="rec_bin")


def get_trailing_edge_dist(ds):
    """
    the lowest elevation at which the signal strength of the waveform is half of the maximum signal above the background noise value
    lowest elevation = largest distance from satellite
    """
    energy_threshold = (ds.rec_wf.max(dim="rec_bin") - ds.noise_mean) / 2.0
    return ds.rec_wf_sample_dist.where(ds.rec_wf >= energy_threshold).max(dim="rec_bin")


def front_slope_to_surface_energy_ratio(ds):
    """
    Front slope to surface energy ratio. We calculated fslope_WHRC as the change in amplitude per meter (volts/meter) in the outer canopy.
    We then applied the following linear transformation in order to calculate fslope on the same scale as provided in data published by
    Margolis et al. (2015): f_slope = 0.5744 + 19.7762 * fslope_WHRC
    """
    # get the highest peak (max amplitude)
    # the fillna is necessary since argmin raises an error otherwise
    # the filled nans will become nans again since canopy_amp at those locations are also nans
    max_ind = ds.processed_wf.fillna(-99).argmax(dim='rec_bin').compute()
    canopy_amp = ds.processed_wf.isel(rec_bin=max_ind)
    canopy_dist = ds.rec_wf_sample_dist.isel(rec_bin=max_ind)

    # calculate amplitude at signal begin as noise mean + nsig * noise sd since this is how signal
    # begin is defined (ie. the highest elevation where signal crosses this threshold)
    # the value of nsig is coded based on the GLAS Algorithm Theoretical Basis Document retrieved at
    # https://www.csr.utexas.edu/glas/pdf/WFAtbd_v5_02011Sept.pdf  (See Appendix 3, pg 99)
    time_of_switch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() + 289742400
    # for any time before the time of switch, b_nsig = 3.5, 7.5 afterwards
    b_nsig = xr.where(ds.time < time_of_switch, x=3.5, y=7.5)

    sig_begin_amp = ds.noise_mean + b_nsig * ds.noise_sd

    # calculate slope as y2-y1 / x2-x1
    fslope_WHRC = (canopy_amp - sig_begin_amp) / (canopy_dist - ds.sig_begin_dist)
    # min max obtained from inspecting data in Margolis et al. (2015)
    return (0.5744 + 19.7762 * fslope_WHRC.clip(min=0)).clip(max=15)


def highest_energy_value(ds):
    """
    Highest energy value in the waveform
    """
    return ds.processed_wf.max(dim="rec_bin")


def wf_variance(ds):
    """
    Variance of the waveform
    """
    weights = ds.processed_wf
    value = ds.rec_wf_sample_dist

    mean = value.weighted(weights).mean(dim="rec_bin")
    sum_of_weights = weights.sum(dim="rec_bin")

    var = (np.square(value - mean) * weights).sum(dim="rec_bin") / sum_of_weights

    return var


def wf_skew(ds):
    """
    Skew of the waveform, directly from GLAH14 data
    """
    return ds.skew


def number_of_peaks(ds):
    """
    Number of peaks found in smoothed waveform, directly from GLAH14 data
    """
    return ds.n_peaks


def proportion_35_to_40m(ds):
    """
    Proportion of the waveform energy from signal beginning to ground peak that is between 35
    and 40 meters in height. Ground peak defined as whichever of the two lowest peaks has greater amplitude using actual wf.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist_actual_wf,
    )

    # then select 35 to 40m
    ht_35_to_40m = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.adj_ground_peak_dist_actual_wf - 40.0,
        signal_end_dist=ds.adj_ground_peak_dist_actual_wf - 35.0,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_ground = sig_beg_to_ground.transpose(dims[0], dims[1])
    ht_35_to_40m = ht_35_to_40m.transpose(dims[0], dims[1])

    return ht_35_to_40m.sum(dim="rec_bin") / sig_beg_to_ground.sum(dim="rec_bin")


def proportion_sig_beg_to_start_of_ground(ds):
    """
    The total energy from signal beginning to the start of the ground peak,
    normalized by total energy of the waveform. Ground peak assumed to be the last peak.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='start_of_ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.modeled_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.start_of_ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.modeled_wf.dims
    sig_beg_to_ground = sig_beg_to_ground.transpose(dims[0], dims[1])

    # total energy of the smoothed waveform
    total = ds.modeled_wf.sum(dim="rec_bin")

    return sig_beg_to_ground.sum(dim="rec_bin") / total


def energy_adj_ground_to_sig_end(ds):
    """
    Waveform energy from the ground peak.  We calculated senergy_whrc as the energy of the waveform (in digital counts) from the ground peak
    to the signal end multiplied by two. Ground peak defined as whichever of the two lowest peaks has greater amplitude. We then applied the
    following linear transformation in order to calculate on the same scale as data published by Margolis et al. (2015)
    senergy = -4.397006 + 0.006208 * senergy_whrc
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    path = 'gs://carbonplan-climatetrace/inputs/volt_table.csv'
    volt_table = pd.read_csv(path)
    volt_to_digital_count = volt_table.set_index('volt_value')['ind'].to_dict()
    wf_in_digital_count = xr.apply_ufunc(
        volt_to_digital_count.__getitem__,
        ds.rec_wf.astype(float).round(6).fillna(-0.195279),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[int],
    )

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist_actual_wf')
    # the processed wf is from sig beg to sig end, select adj ground peak to sig end instead
    ground_energy = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=wf_in_digital_count,
        signal_begin_dist=ds.adj_ground_peak_dist_actual_wf,
        signal_end_dist=ds.sig_end_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    ground_energy = ground_energy.transpose(dims[0], dims[1])

    senergy_whrc = ground_energy.sum(dim="rec_bin") * 2

    return -4.397006 + 0.006208 * senergy_whrc


def all_zero_variable(ds, template_var='sig_begin_dist'):
    """
    Returns all zeros in a dataarray in the shape of the template var in terms of record index and shot number
    """
    return xr.DataArray(
        0,
        dims=["unique_index"],
        coords=[ds[template_var].coords["unique_index"]],
    )


def acq3_Neigh(ds):
    """
    Dummy variable used by Neigh et al. (2013) identifying shots from laser campaign L3F.
    Campaign dates retrieved from https://nsidc.org/data/icesat/orbit_grnd_trck.html
    """
    output = all_zero_variable(ds)

    # campaign L3F is between 2006-05-24 2006-06-26
    begin = datetime(2006, 5, 24, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end = datetime(2006, 6, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # set values within the dates to be 1
    return output.where(((ds.time >= begin) & (ds.time <= end)), other=1)


def acq2_Nelson(ds):
    """
    Dummy variable used by Nelson et al. (2017) identifying shots from laser campaign L3F.
    Campaign dates retrieved from https://nsidc.org/data/icesat/orbit_grnd_trck.html
    """
    return acq3_Neigh(ds)


def acq3_Nelson(ds):
    """
    Dummy variable used by Nelson et al. (2017) identifying shots from laser campaign L3D.
    Campaign dates retrieved from https://nsidc.org/data/icesat/orbit_grnd_trck.html
    """
    output = all_zero_variable(ds)

    # campaign L3D is between 2005-10-21 2005-11-24
    begin = datetime(2005, 10, 21, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end = datetime(2005, 11, 24, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    # set values within the dates to be 1
    return output.where(((ds.time >= begin) & (ds.time <= end)), other=1)


def plot_shot(record):
    from carbonplan_trace.v1.glas_preprocess import SECONDS_IN_NANOSECONDS, SPEED_OF_LIGHT, gaussian

    bins = record.rec_wf_sample_dist
    plt.figure(figsize=(6, 10))
    plt.scatter(record.rec_wf, bins, s=5, label="Raw")  # raw wf
    xmax = record.rec_wf.max(dim='rec_bin') + 0.1

    gaussian_sigma_in_m = (record.gaussian_sigma * SECONDS_IN_NANOSECONDS * SPEED_OF_LIGHT) / 2
    gaussians = gaussian(
        x=record['rec_wf_sample_dist'],
        amplitude=record['gaussian_amp'],
        mean=record['gaussian_fit_dist'],
        stddev=gaussian_sigma_in_m,
    )
    for i, g in enumerate(gaussians):
        if i == 0:
            plt.plot(g, bins, '0.5', label='gaussin curve')
        else:
            plt.plot(g, bins, '0.5')

    plt.plot(record['modeled_wf'], bins, 'r-', label='gaussian summed')

    # plot various variables found in GLAH14
    plt.plot(
        [-0.05, xmax],
        np.array([record.sig_begin_dist, record.sig_begin_dist]),
        "g--",
        label="Signal Beginning",
    )
    plt.plot(
        [-0.05, xmax],
        np.array([record.sig_end_dist, record.sig_end_dist]),
        "g--",
        label="Signal End",
    )

    # plot noise mean and std from GLAH01
    plt.plot(
        [record.noise_mean, record.noise_mean], [bins.min(), bins.max()], "0.5", label="Noise Mean"
    )
    n_sig = 3.5
    noise_threshold = record.noise_mean + n_sig * record.noise_sd
    plt.plot(
        [noise_threshold, noise_threshold],
        [bins.min(), bins.max()],
        color="0.5",
        linestyle="dashed",
        label="Noise Threshold",
    )

    # plot filtered wf
    plt.plot(record.processed_wf + record.noise_mean, bins, "k-", label="Filtered Waveform")

    plt.plot(
        [-0.05, xmax],
        [record.ground_peak_dist, record.ground_peak_dist],
        "y--",
        label="Ground Peak",
    )

    plt.plot(
        [-0.05, xmax],
        [record.adj_ground_peak_dist_actual_wf, record.adj_ground_peak_dist_actual_wf],
        "c--",
        label="Ground (Actual)",
    )

    # plt.ylim(record.sig_end_dist+10, record.sig_begin_dist-10)
    plt.gca().invert_yaxis()
    plt.xlabel("lidar return (joules)")
    plt.ylabel("distance from satelite (m)")
    plt.legend()
    plt.show()
    plt.close()


DISTANCE_METRICS_MAP = {
    "sig_begin_dist": get_sig_begin_dist,
    "sig_end_dist": get_sig_end_dist,
    "centroid_dist": get_centroid_dist,
    "gaussian_fit_dist": get_gaussian_fit_dist,
    "ground_peak_dist": get_ground_peak_dist,
    "first_peak_dist": get_first_peak_dist,
    "adj_ground_peak_dist": get_adj_ground_peak_dist,
    "adj_ground_peak_dist_actual_wf": get_adj_ground_peak_dist_actual_wf,
    "quadratic_mean_dist": get_quadratic_mean_dist,
    "mean_dist": get_mean_dist,
    "pct_05_dist": partial(get_percentile_dist, percentile=5),
    "pct_10_dist": partial(get_percentile_dist, percentile=10),
    "pct_20_dist": partial(get_percentile_dist, percentile=20),
    "pct_25_dist": partial(get_percentile_dist, percentile=25),
    "pct_30_dist": partial(get_percentile_dist, percentile=30),
    "pct_40_dist": partial(get_percentile_dist, percentile=40),
    "pct_50_dist": partial(get_percentile_dist, percentile=50),
    "pct_75_dist": partial(get_percentile_dist, percentile=75),
    "pct_80_dist": partial(get_percentile_dist, percentile=80),
    "pct_90_dist": partial(get_percentile_dist, percentile=90),
    "pct_90_dist_modeled_wf": partial(get_percentile_dist_modeled_wf, percentile=90),
    "pct_75_dist_modeled_wf": partial(get_percentile_dist_modeled_wf, percentile=75),
    "pct_50_dist_modeled_wf": partial(get_percentile_dist_modeled_wf, percentile=50),
    "pct_40_dist_modeled_wf": partial(get_percentile_dist_modeled_wf, percentile=40),
    "pct_10_dist_modeled_wf": partial(get_percentile_dist_modeled_wf, percentile=10),
    "wf_max_e_dist": get_wf_max_e_dist,
    "start_of_ground_peak_dist": get_start_of_ground_peak_dist,
    "leading_edge_dist": get_leading_edge_dist,
    "trailing_edge_dist": get_trailing_edge_dist,
    "pct_10_of_sig_beg_and_adj_ground_dist_actual_wf": partial(
        get_percentile_of_sig_beg_and_adj_ground_dist_actual_wf, percentile=10
    ),
    "pct_80_of_sig_beg_and_adj_ground_dist_actual_wf": partial(
        get_percentile_of_sig_beg_and_adj_ground_dist_actual_wf, percentile=80
    ),
    "pct_90_of_sig_beg_and_adj_ground_dist_actual_wf": partial(
        get_percentile_of_sig_beg_and_adj_ground_dist_actual_wf, percentile=90
    ),
    "bottom_of_canopy_dist": get_bottom_of_canopy_dist,
    "pct_05_canopy_dist": partial(get_percentile_canopy_dist, percentile=5),
    "pct_10_canopy_dist": partial(get_percentile_canopy_dist, percentile=10),
    "pct_20_canopy_dist": partial(get_percentile_canopy_dist, percentile=20),
    "pct_30_canopy_dist": partial(get_percentile_canopy_dist, percentile=30),
    "pct_50_canopy_dist": partial(get_percentile_canopy_dist, percentile=50),
    "pct_75_canopy_dist": partial(get_percentile_canopy_dist, percentile=75),
    "pct_80_canopy_dist": partial(get_percentile_canopy_dist, percentile=80),
}


def get_dist_metric_value(ds, metric):
    if metric not in ds and metric in DISTANCE_METRICS_MAP:
        ds[metric] = DISTANCE_METRICS_MAP[metric](ds)
    elif metric not in ds:
        raise NotImplementedError(
            f'Metric {metric} not found in dataset and not included in DISTANCE_METRICS_MAP for calculation'
        )

    return ds


def get_height_metric_value(ds, metric, recalc=False):
    if (metric not in ds and metric in HEIGHT_METRICS_MAP) or recalc:
        ds[metric] = HEIGHT_METRICS_MAP[metric](ds)
    elif metric not in ds:
        raise NotImplementedError(
            f'Metric {metric} not found in dataset and not included in HEIGHT_METRICS_MAP for calculation'
        )

    return ds


def get_all_height_metrics(ds, metrics, recalc=False):
    for metric in metrics:
        ds = get_height_metric_value(ds, metric, recalc=recalc)

    return ds


def get_heights_from_distance(ds, top_metric, bottom_metric):
    # check if the metric is in input ds, recalculate if not
    for metric in [top_metric, bottom_metric]:
        ds = get_dist_metric_value(ds, metric)

    # multiply with -1 since distance is measured from satellite to object, thus top has a smaller value
    return -1 * (ds[top_metric] - ds[bottom_metric])


HEIGHT_METRICS_MAP = {
    "VH": sig_beg_to_ground_ht,
    "H10_Baccini": pct_10_from_sig_beg_to_sig_end_ht,
    "H25_Baccini": pct_25_from_sig_beg_to_sig_end_ht,
    "H60_Baccini": pct_60_from_sig_beg_to_sig_end_ht,
    "CANOPY_DEP": sig_beg_to_start_of_ground_peak_ht,
    "CANOPY_ENE": proportion_sig_beg_to_start_of_ground,
    "ht_adjusted": sig_beg_to_adj_ground_ht,
    "QMCH": quadratic_mean_to_adj_ground_peak_actual_wf_ht,
    "MeanH": mean_to_adj_ground_peak_actual_wf_ht,
    "HOME_Baccini": pct_50_from_sig_beg_to_sig_end_ht,
    "HOME_Yavasli": centroid_to_adj_ground_ht,
    "pct_HOME_Yavasli": ratio_centroid_to_max_ht,
    "h25_Neigh": pct_25_to_adj_ground_peak_actual_wf_ht,
    "h50_Neigh": pct_50_to_adj_ground_peak_actual_wf_ht,
    "h75_Neigh": pct_75_to_adj_ground_peak_actual_wf_ht,
    "h90_Neigh": pct_90_to_adj_ground_peak_actual_wf_ht,
    "h05_Nelson": pct_05_to_sig_end_ht,
    "h10_Nelson": pct_10_to_sig_end_ht,
    "h20_Nelson": pct_20_to_sig_end_ht,
    "h25_Nelson": pct_25_to_sig_end_ht,
    "h30_Nelson": pct_30_to_sig_end_ht,
    "h50_Nelson": pct_50_to_sig_end_ht,
    "h75_Nelson": pct_75_to_sig_end_ht,
    "h80_Nelson": pct_80_to_sig_end_ht,
    "h90_Nelson": pct_90_to_sig_end_ht,
    "h10_p12": pct_10_of_sig_beg_and_adj_ground_actual_wf_ht,
    "h80_p12": pct_80_of_sig_beg_and_adj_ground_actual_wf_ht,
    "h90_p12": pct_90_of_sig_beg_and_adj_ground_actual_wf_ht,
    "h05_canopy": pct_05_canopy_ht,
    "h10_canopy": pct_10_canopy_ht,
    "h20_canopy": pct_20_canopy_ht,
    "h30_canopy": pct_30_canopy_ht,
    "h50_canopy": pct_50_canopy_ht,
    "h75_canopy": pct_75_canopy_ht,
    "h80_canopy": pct_80_canopy_ht,
    "f_slope": front_slope_to_surface_energy_ratio,
    "senergy": energy_adj_ground_to_sig_end,
    "trail_Nelson": adj_ground_to_sig_end_ht,
    "lead_Nelson": start_to_centroid_adj_ground_ht,
    "Height_35_to_40": proportion_35_to_40m,
    "acq3_Neigh": acq3_Neigh,
    "acq2_Nelson": acq2_Nelson,
    "acq3_Nelson": acq3_Nelson,
    "wf_extent": sig_beg_to_sig_end_ht,
    "leading_edge_extent": get_leading_edge_extent,
    "trailing_edge_extent": get_trailing_edge_extent,
    # "wf_max_e": highest_energy_value,
    # "wf_variance": wf_variance,
    # "wf_skew": wf_skew,
    # "startpeak": sig_beg_to_highest_energy_ht,
    # "wf_n_gs": number_of_peaks,
}
