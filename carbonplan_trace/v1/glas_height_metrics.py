from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def sig_beg_to_sig_end_ht(ds):
    """
    Waveform extent from signal beginning to end
    """
    return get_heights_from_distance(ds, top_metric='sig_begin_dist', bottom_metric='sig_end_dist')


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


def first_peak_to_adj_ground_ht(ds):
    """
    Vertical distance, in meters, between the height of the ground peak and the height of the first peak
    (i.e., the peak closest to the satellite). Ground peak defined as whichever of the two lowest peaks
    has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='first_peak_dist', bottom_metric='adj_ground_peak_dist'
    )


def quadratic_mean_to_adj_ground_ht(ds):
    """
    Quadratic mean height of the waveform from ground peak to signal beginning (meters).
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='quadratic_mean_dist', bottom_metric='adj_ground_peak_dist'
    )


def mean_to_adj_ground_ht(ds):
    """
    Mean height of the waveform from ground peak to signal beginning (meters).
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='mean_dist', bottom_metric='adj_ground_peak_dist'
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


def ground_to_sig_end_ht(ds):
    """
    Distance, in meters, from the ground peak to the signal end. Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='ground_peak_dist', bottom_metric='sig_end_dist'
    )


def sig_beg_to_highest_energy_ht(ds):
    """
    Distance between the height of signal beginning and the height of wf_max_e
    """
    return get_heights_from_distance(ds, top_metric='sig_begin_dist', bottom_metric='wf_max_e_dist')


def start_to_centroid_ground_ht(ds):
    """
    Distance, in meters, from the start of the ground signal to the ground peak.
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='adj_ground_peak_dist'
    )


def sig_beg_to_start_of_ground_peak_ht(ds):
    """
    The distance from the signal beginning to the start of the ground peak (meters). Ground peak assumed to be the last peak.
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='start_of_ground_peak_dist'
    )


# TODO: change the next set of functions to take in pct as a param and avoid repeating
def pct_25_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_25_dist', bottom_metric='adj_ground_peak_dist'
    )


def pct_50_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_50_dist', bottom_metric='adj_ground_peak_dist'
    )


def pct_75_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_75_dist', bottom_metric='adj_ground_peak_dist'
    )


def pct_90_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_90_dist', bottom_metric='adj_ground_peak_dist'
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
    return get_heights_from_distance(ds, top_metric='pct_90_dist', bottom_metric='sig_end_dist')


def pct_25_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_75_dist', bottom_metric='sig_end_dist')


def pct_60_from_sig_beg_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_40_dist', bottom_metric='sig_end_dist')


def pct_10_of_sig_beg_and_adj_ground_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_10_of_sig_beg_and_adj_ground_dist', bottom_metric='adj_ground_peak_dist'
    )


def pct_80_of_sig_beg_and_adj_ground_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_80_of_sig_beg_and_adj_ground_dist', bottom_metric='adj_ground_peak_dist'
    )


def pct_90_of_sig_beg_and_adj_ground_to_adj_ground_ht(ds):
    return get_heights_from_distance(
        ds, top_metric='pct_90_of_sig_beg_and_adj_ground_dist', bottom_metric='adj_ground_peak_dist'
    )


def get_leading_edge_extent(ds):
    """
    the height difference between the elevation of the signal start and the first elevation at which the
    waveform is half of the maximum signal above the background noise value.
    see https://daac.ornl.gov/NACP/guides/NACP_Boreal_Biome_Biomass.html for more details
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='leading_edge_dist'
    )


def get_trailing_edge_extent(ds):
    """
    the height difference between the lowest elevation at which the signal strength of the waveform is half of
    the maximum signal above the background noise value, and the elevation of the signal end
    see https://daac.ornl.gov/NACP/guides/NACP_Boreal_Biome_Biomass.html for more details
    """
    return get_heights_from_distance(
        ds, top_metric='trailing_edge_dist', bottom_metric='sig_end_dist'
    )


def get_sig_begin_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    ref_range_bias = ds.rec_wf_sample_dist.max(dim="rec_bin") - ds.ref_range
    return ds.sig_begin_offset + ds.ref_range + ref_range_bias


def get_sig_end_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    ref_range_bias = ds.rec_wf_sample_dist.max(dim="rec_bin") - ds.ref_range
    return ds.sig_end_offset + ds.ref_range + ref_range_bias


def get_centroid_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    ref_range_bias = ds.rec_wf_sample_dist.max(dim="rec_bin") - ds.ref_range
    return ds.centroid_offset + ds.ref_range + ref_range_bias


def get_gaussian_fit_dist(ds):
    """
    Obtained directly from GLAH14 data
    """
    # calculate the bias between reference range to the bottom of received wf
    ref_range_bias = ds.rec_wf_sample_dist.max(dim="rec_bin") - ds.ref_range
    return ds.gaussian_mu + ds.ref_range + ref_range_bias


def get_percentile_dist(ds, percentile):
    """
    the distance at which x% of the total waveform energy from signal beginning to signal end has been reached.
    energy accumulated relative to signal end.
    as an example, 25th percentile dist should be larger than 50th percentile dist, where larger dist = further
    to the satellite = lower elevation on earth
    percentiles is a list in hundredth format (e.g. to get 10th percentile input value 10)
    """
    total = ds['processed_wf'].sum(dim="rec_bin")
    cumsum = ds['processed_wf'].cumsum(dim="rec_bin")
    target = total * percentile / 100.0

    return ds.rec_wf_sample_dist.where(cumsum > target).max(dim="rec_bin")


def get_percentile_of_sig_beg_and_adj_ground_dist(ds, percentile):
    """
    the distance, in meters, between the ground peak and the height at which 10% of the waveform energy
    from ground peak to signal beginning has been reached. Waveform energy accumulation relative to
     ground peak. Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    total = sig_beg_to_adj_ground.sum(dim="rec_bin")
    cumsum = sig_beg_to_adj_ground.cumsum(dim="rec_bin")
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


def get_ground_peak_dist_Sun(ds, buffer=1):
    """
    Identify lowest peak in smoothed waveform, adopted after Sun et al 2008.
    buffer indicates the lowest bin that can be identified as ground peak (buffer = 3 indicates that
    the lowest 3 bins are excluded from ground peak identification)
    """
    assert buffer >= 1

    # ensure that things are ordered the same way
    all_distances = ds.rec_wf_sample_dist.transpose("rec_bin", "record_index", "shot_number")
    wf = ds.processed_wf.transpose("rec_bin", "record_index", "shot_number")

    # initialize an array of ground peak distance with the shape of record index x shot number
    ground_distance = xr.DataArray(
        0,
        dims=["record_index", "shot_number"],
        coords=[wf.coords["record_index"], wf.coords["shot_number"]],
    )

    for i in np.arange(buffer, wf.rec_bin.shape[0] - 1):
        mask = (
            # where the current bin has waveform intensity larger then the previous bin and the next bin
            (wf.isel(rec_bin=i) > wf.isel(rec_bin=i - 1))
            & (wf.isel(rec_bin=i) > wf.isel(rec_bin=i + 1))
            & (ground_distance == 0)  # and this is the first peak found
        )

        # where mask = True, set the ground distance to be equal to distance of current bin i
        # otherwise continue to use the data stored in ground distance
        ground_distance = xr.where(mask, x=all_distances.isel(rec_bin=i), y=ground_distance)

    # set the 0s (records where we didn't find a peak) in distance to the max distance (bin 0)
    mask = ground_distance == 0
    ground_distance = xr.where(mask, all_distances.isel(rec_bin=0), ground_distance)

    return ground_distance


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
    loc = ds.gaussian_amp.isel(n_gaussian_peaks=slice(2)).fillna(0).argmax(dim="n_gaussian_peaks")
    return ds.gaussian_fit_dist.isel(n_gaussian_peaks=loc)


def get_quadratic_mean_dist(ds):
    """
    From Neigh et al 2013: quadratic mean height of the waveform, calculated as the
    square root [∑ (normalized amplitude in a given canopy height bin) × (height of bin)**2]
    Original citation from Lefsky et al 1999. From signal beginning to adj ground peak.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    bins = ds.rec_wf_sample_dist

    return np.sqrt(np.square(bins).weighted(sig_beg_to_adj_ground).mean("rec_bin"))


def get_mean_dist(ds):
    """
    Mean height of the waveform from the signal beginning to adj ground peak
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_adj_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_adj_ground = sig_beg_to_adj_ground.transpose(dims[0], dims[1])

    bins = ds.rec_wf_sample_dist

    return bins.weighted(sig_beg_to_adj_ground).mean("rec_bin")


# TODO: change the next set of functions to take in pct as a param and avoid repeating
def get_05th_pct_dist(ds):
    return get_percentile_dist(ds, 5)


def get_10th_pct_dist(ds):
    return get_percentile_dist(ds, 10)


def get_20th_pct_dist(ds):
    return get_percentile_dist(ds, 20)


def get_25th_pct_dist(ds):
    return get_percentile_dist(ds, 25)


def get_30th_pct_dist(ds):
    return get_percentile_dist(ds, 30)


def get_40th_pct_dist(ds):
    return get_percentile_dist(ds, 40)


def get_50th_pct_dist(ds):
    return get_percentile_dist(ds, 50)


def get_75th_pct_dist(ds):
    return get_percentile_dist(ds, 75)


def get_80th_pct_dist(ds):
    return get_percentile_dist(ds, 80)


def get_90th_pct_dist(ds):
    return get_percentile_dist(ds, 90)


def get_pct_10_of_sig_beg_and_adj_ground_dist(ds):
    return get_percentile_of_sig_beg_and_adj_ground_dist(ds, 10)


def get_pct_80_of_sig_beg_and_adj_ground_dist(ds):
    return get_percentile_of_sig_beg_and_adj_ground_dist(ds, 80)


def get_pct_90_of_sig_beg_and_adj_ground_dist(ds):
    return get_percentile_of_sig_beg_and_adj_ground_dist(ds, 90)


def get_wf_max_e_dist(ds):
    """
    distance at which max waveform energy occurs
    """
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_dist

    return bins.isel(rec_bin=wf.argmax(dim="rec_bin"))


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
    Margolis et al. (2015): f_slope  =0.5744 + 19.7762⋅fslope_WHRC
    """
    # get the highest peak (highest in elevation = smallest distance)
    # the fillna is necessary since argmin raises an error otherwise
    # the filled nans will become nans again since canopy_amp at those locations are also nans
    canopy_dist = ds.gaussian_fit_dist.fillna(1e10).argmin(dim="n_gaussian_peaks")
    canopy_amp = ds.gaussian_amp.isel(n_gaussian_peaks=canopy_dist)

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
    return 0.5744 + 19.7762 * fslope_WHRC


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
    and 40 meters in height. Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_ground = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.adj_ground_peak_dist,
    )

    # then select 35 to 40m
    ht_35_to_40m = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.adj_ground_peak_dist - 40.0,
        signal_end_dist=ds.adj_ground_peak_dist - 35.0,
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
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.start_of_ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_ground = sig_beg_to_ground.transpose(dims[0], dims[1])

    # total energy of the smoothed waveform
    total = ds.processed_wf.sum(dim="rec_bin")

    return sig_beg_to_ground.sum(dim="rec_bin") / total


def pct_canopy_cover(ds, cutoff_height=1.0):
    """
    Mean percent canopy cover (range 0-100).
    Cover was computed by dividing returns above the cover height threshold by total
    number of returns (including those below the ground height cut-off) that were in the plot.
    cutoff_height is in meters, return values are in % (0-100)
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak + cutoff height
    vegetation_returns = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.sig_begin_dist,
        signal_end_dist=ds.ground_peak_dist - cutoff_height,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    vegetation_returns = vegetation_returns.transpose(dims[0], dims[1])

    # normalized by the total energy between signal beginning and end
    return vegetation_returns.sum(dim="rec_bin") / ds.processed_wf.sum(dim="rec_bin") * 100


def energy_adj_ground_to_sig_end(ds):
    """
    Waveform energy from the ground peak.  We calculated senergy_whrc as the energy of the waveform (in digital counts) from the ground peak
    to the signal end multiplied by two. Ground peak defined as whichever of the two lowest peaks has greater amplitude. We then applied the
    following linear transformation in order to calculate on the same scale as data published by Margolis et al. (2015)
    senergy = -4.397006 + 0.006208 * senergy_whrc
    """
    from carbonplan_trace.v1.glas_preprocess import select_valid_area  # avoid circular import

    ds = get_dist_metric_value(ds, metric='adj_ground_peak_dist')
    # the processed wf is from sig beg to sig end, select adj ground peak to sig end instead
    ground_energy = select_valid_area(
        bins=ds.rec_wf_sample_dist,
        wf=ds.processed_wf,
        signal_begin_dist=ds.adj_ground_peak_dist,
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


def terrain_relief_class_1(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_1=elev_diff≥0 m and elev_diff<7 m.
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='sig_end_dist'
    )
    return xr.where((relief >= 0) & (relief < 7), x=1, y=0)


def terrain_relief_class_2(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_2=elev_diff≥7 m and elev_diff<15 m.
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='sig_end_dist'
    )
    return xr.where((relief >= 7) & (relief < 15), x=1, y=0)


def terrain_relief_class_3(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_3=elev_diff≥15 m
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='sig_end_dist'
    )
    return xr.where(relief >= 15, x=1, y=0)


def plot_shot(record):
    bins = record.rec_wf_sample_dist
    plt.figure(figsize=(6, 10))
    plt.scatter(record.rec_wf, bins, s=5, label="Raw")  # raw wf

    # plot various variables found in GLAH14
    plt.plot(
        [-0.05, 0.5],
        np.array([record.sig_begin_dist, record.sig_begin_dist]),
        "r--",
        label="Signal Beginning",
    )
    plt.plot(
        [-0.05, 0.5],
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

    # plot percentile heights
    plt.plot(
        [-0.05, 0.5],
        [record["10th_distance"], record["10th_distance"]],
        "b--",
        label="10th Percentile",
    )
    plt.plot([-0.05, 0.5], [record.meanH_dist, record.meanH_dist], "c--", label="Mean H")
    plt.plot(
        [-0.05, 0.5],
        [record["90th_distance"], record["90th_distance"]],
        "m--",
        label="90th Percentile",
    )
    plt.plot(
        [-0.05, 0.5],
        [record.ground_peak_dist, record.ground_peak_dist],
        "y--",
        label="Ground Peak",
    )

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
    "quadratic_mean_dist": get_quadratic_mean_dist,
    "mean_dist": get_mean_dist,
    "pct_05_dist": get_05th_pct_dist,
    "pct_10_dist": get_10th_pct_dist,
    "pct_20_dist": get_20th_pct_dist,
    "pct_25_dist": get_25th_pct_dist,
    "pct_30_dist": get_30th_pct_dist,
    "pct_40_dist": get_40th_pct_dist,
    "pct_50_dist": get_50th_pct_dist,
    "pct_75_dist": get_75th_pct_dist,
    "pct_80_dist": get_80th_pct_dist,
    "pct_90_dist": get_90th_pct_dist,
    "wf_max_e_dist": get_wf_max_e_dist,
    "start_of_ground_peak_dist": get_start_of_ground_peak_dist,
    "leading_edge_dist": get_leading_edge_dist,
    "trailing_edge_dist": get_trailing_edge_dist,
    "pct_10_of_sig_beg_and_adj_ground_dist": get_pct_10_of_sig_beg_and_adj_ground_dist,
    "pct_80_of_sig_beg_and_adj_ground_dist": get_pct_80_of_sig_beg_and_adj_ground_dist,
    "pct_90_of_sig_beg_and_adj_ground_dist": get_pct_90_of_sig_beg_and_adj_ground_dist,
}


def get_dist_metric_value(ds, metric):
    if metric not in ds and metric in DISTANCE_METRICS_MAP:
        ds[metric] = DISTANCE_METRICS_MAP[metric](ds)
    elif metric not in ds:
        raise NotImplementedError(
            f'Metric {metric} not found in dataset and not included in DISTANCE_METRICS_MAP for calculation'
        )

    return ds


def get_height_metric_value(ds, metric):
    if metric not in ds and metric in HEIGHT_METRICS_MAP:
        ds[metric] = HEIGHT_METRICS_MAP[metric](ds)
    elif metric not in ds:
        raise NotImplementedError(
            f'Metric {metric} not found in dataset and not included in HEIGHT_METRICS_MAP for calculation'
        )

    return ds


def get_all_height_metrics(ds, metrics):
    for metric in metrics:
        ds = get_height_metric_value(ds, metric)

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
    "ht_adjusted": first_peak_to_adj_ground_ht,
    "QMCH": quadratic_mean_to_adj_ground_ht,
    "MeanH": mean_to_adj_ground_ht,
    "HOME_Baccini": pct_50_to_sig_end_ht,
    "HOME_Yavasli": centroid_to_adj_ground_ht,
    "pct_HOME_Yavasli": ratio_centroid_to_max_ht,
    "h25_Neigh": pct_25_to_adj_ground_ht,
    "h50_Neigh": pct_50_to_adj_ground_ht,
    "h75_Neigh": pct_75_to_adj_ground_ht,
    "h90_Neigh": pct_90_to_adj_ground_ht,
    "h05_Nelson": pct_05_to_sig_end_ht,
    "h10_Nelson": pct_10_to_sig_end_ht,
    "h20_Nelson": pct_20_to_sig_end_ht,
    "h25_Nelson": pct_25_to_sig_end_ht,
    "h30_Nelson": pct_30_to_sig_end_ht,
    "h50_Nelson": pct_50_to_sig_end_ht,
    "h75_Nelson": pct_75_to_sig_end_ht,
    "h80_Nelson": pct_80_to_sig_end_ht,
    "h90_Nelson": pct_90_to_sig_end_ht,
    "h10_p12": pct_10_of_sig_beg_and_adj_ground_to_adj_ground_ht,
    "h80_p12": pct_80_of_sig_beg_and_adj_ground_to_adj_ground_ht,
    "h90_p12": pct_90_of_sig_beg_and_adj_ground_to_adj_ground_ht,
    "f_slope": front_slope_to_surface_energy_ratio,
    "senergy": energy_adj_ground_to_sig_end,
    "trail_Nelson": ground_to_sig_end_ht,
    "lead_Nelson": start_to_centroid_ground_ht,
    "Height_35_to_40": proportion_35_to_40m,
    "treecover2000_mean": pct_canopy_cover,  # TODO: use additional dataset instead
    "acq3_Neigh": all_zero_variable,
    "acq2_Nelson": all_zero_variable,
    "acq3_Nelson": all_zero_variable,
    "ct2_Nelson": all_zero_variable,
    # "wf_max_e": highest_energy_value,
    # "wf_variance": wf_variance,
    # "wf_skew": wf_skew,
    # "startpeak": sig_beg_to_highest_energy_ht,
    # "wf_n_gs": number_of_peaks,
    # "terrain_1": terrain_relief_class_1,
    # "terrain_2": terrain_relief_class_2,
    # "terrain_3": terrain_relief_class_3,
}
