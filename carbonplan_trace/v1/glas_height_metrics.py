import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from carbonplan_trace.v1.glas_preprocess import select_valid_area


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


def quadratic_mean_to_ground_ht(ds):
    """
    Quadratic mean height of the waveform from the signal beginning to the last peak (meters).
    """
    return get_heights_from_distance(
        ds, top_metric='quadratic_mean_dist', bottom_metric='ground_peak_dist'
    )


def mean_to_ground_ht(ds):
    """
    Mean height of the waveform from the signal beginning to the last peak (meters).
    """
    return get_heights_from_distance(ds, top_metric='mean_dist', bottom_metric='ground_peak_dist')


def median_to_sig_end_ht(ds):
    """
    Height of median energy (meters). The distance, in meters, between the height at which 50% of the total waveform energy from signal
    beginning to signal end has been reached and the height of signal end.
    """
    return get_heights_from_distance(ds, top_metric='pct_50_dist', bottom_metric='sig_end_dist')


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
    Distance, in meters, from the start of the ground signal to the ground peak.
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.
    """
    return get_heights_from_distance(
        ds, top_metric='start_of_adj_ground_peak_dist', bottom_metric='adj_ground_peak_dist'
    )


def sig_beg_to_start_of_ground_peak_ht(ds):
    """
    The distance from the signal beginning to the start of the ground peak (meters). Ground peak assumed to be the last peak.
    """
    return get_heights_from_distance(
        ds, top_metric='sig_begin_dist', bottom_metric='start_of_ground_peak_dist'
    )


def pct_25_to_ground_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_25_dist', bottom_metric='ground_peak_dist')


def pct_50_to_ground_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_50_dist', bottom_metric='ground_peak_dist')


def pct_75_to_ground_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_75_dist', bottom_metric='ground_peak_dist')


def pct_90_to_ground_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_90_dist', bottom_metric='ground_peak_dist')


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


def pct_60_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_60_dist', bottom_metric='sig_end_dist')


def pct_75_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_75_dist', bottom_metric='sig_end_dist')


def pct_80_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_80_dist', bottom_metric='sig_end_dist')


def pct_90_to_sig_end_ht(ds):
    return get_heights_from_distance(ds, top_metric='pct_90_dist', bottom_metric='sig_end_dist')


def get_percentile_dist(ds, percentile):
    """
    bins and wf are expected to be data arrays with rec_bin, record_index, shot_number as dims
    percentiles is a list in hundredth format (e.g. to get 10th percentile input value 10)
    """
    total = ds.processed_wf.sum(dim="rec_bin")
    cumsum = ds.processed_wf.cumsum(dim="rec_bin")
    target = total * percentile / 100.0

    return ds.rec_wf_sample_distance.where(cumsum > target).max(dim="rec_bin")


def get_ground_peak_dist(ds, buffer=1):
    """
    bins and wf are expected to be data arrays with rec_bin, record_index, shot_number as dims
    buffer is the minimum index where the ground peak can be located
    """
    # TODO: just use the GLAH14 definition?
    assert buffer >= 1

    # ensure that things are ordered the same way
    bins = ds.rec_wf_sample_distance.transpose("rec_bin", "record_index", "shot_number")
    wf = ds.processed_wf.transpose("rec_bin", "record_index", "shot_number")

    # initialize an array of ground peak distance with the shape of record index x shot number
    distance = xr.DataArray(
        0,
        dims=["record_index", "shot_number"],
        coords=[wf.coords["record_index"], wf.coords["shot_number"]],
    )

    for i in np.arange(buffer, wf.rec_bin.shape[0] - 1):
        mask = (
            # where the current bin has waveform intensity larger then the previous bin and the next bin
            (wf.isel(rec_bin=i) > wf.isel(rec_bin=i - 1))
            & (wf.isel(rec_bin=i) > wf.isel(rec_bin=i + 1))
            & (distance == 0)  # and this is the first peak found
        )

        distance = xr.where(mask, x=bins.isel(rec_bin=i), y=distance)

    # set the 0 in distance to the max distance in wf
    mask = distance == 0
    distance = xr.where(mask, bins.isel(rec_bin=0), distance)

    return distance


def get_start_of_ground_peak_dist(ds):
    pass


def get_end_of_ground_peak_dist(ds):
    pass


def get_adj_ground_peak_dist(ds):
    # compare the amplitude of the last two peaks and mark the greater one
    # get the corresponding distance of the gaussian fit
    # return this distance
    pass


def get_start_of_adj_ground_peak_dist(ds):
    # find the adjusted ground peak
    # find the distance at which gaussian signal would be lower than noise based on amp, sigma, and loation
    # get diff between the acutal peak and the signal start
    pass


def get_quadratic_mean_dist(ds):
    """
    From Neigh et al 2013: quadratic mean height of the waveform, calculated as the
    square root [∑ (normalized amplitude in a given canopy height bin) × (height of bin)**2]
    Original citation from Lefsky et al 1999
    """
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_distance

    return bins.square().weighted(wf).mean("rec_bin").sqrt()


def get_mean_dist(ds):
    """
    Mean height of the waveform from the signal beginning to signal end
    """
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_distance

    return bins.weighted(wf).mean("rec_bin")


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


def get_50th_pct_dist(ds):
    return get_percentile_dist(ds, 50)


def get_60th_pct_dist(ds):
    return get_percentile_dist(ds, 60)


def get_75th_pct_dist(ds):
    return get_percentile_dist(ds, 75)


def get_80th_pct_dist(ds):
    return get_percentile_dist(ds, 80)


def get_90th_pct_dist(ds):
    return get_percentile_dist(ds, 90)


def get_wf_max_e_dist(ds):
    """
    distance at which max waveform energy occurs
    """
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_distance

    return bins.isel(wf.argmax(dim=["rec_bin"]))


def front_slope_to_surface_energy_ratio(ds):
    """
    Front slope to surface energy ratio. We calculated fslope_WHRC as the change in amplitude per meter (volts/meter) in the outer canopy.
    We then applied the following linear transformation in order to calculate fslope on the same scale as provided in data published by
    Margolis et al. (2015): f_slope  =0.5744 + 19.7762⋅fslope_WHRC
    """
    # find canopy_peak dist = distance of highest (in terms of distance from ground) Gaussian peak
    # find canopy_peak amp = amplitude of highest Gaussian peak - signal noise  --> may not work and need to find the first peak manually
    # find sig beg dist
    # find amplitude of the sig beg = 0
    # calculate slope as y2-y1 / x2-x1
    fslope_WHRC = 0
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
    wf = ds.processed_wf
    bins = ds.rec_wf_sample_distance

    return bins.weighted(wf).var(dim="rec_bin")


def wf_skew(ds):
    """
    Skew of the waveform
    """
    return ds.skew


def number_of_gaussian_peaks(ds):
    """
    Number of Gaussian curves found in waveform
    """
    return ds.num_gaussian_peaks


def proportion_35_to_40m(ds):
    """
    Proportion of the waveform energy from signal beginning to ground peak that is between 35
    and 40 meters in height. Ground peak assumed to be the last peak.
    """
    ds = get_metric_value(ds, metric='ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_ground = select_valid_area(
        bins=ds.rec_wf_sample_distance,
        wf=ds.processed_wf,
        signal_begin_distance=ds.sig_begin_dist,
        signal_end_distance=ds.ground_peak_dist,
    )

    # then select 35 to 40m
    ht_35_to_40m = select_valid_area(
        bins=ds.rec_wf_sample_distance,
        wf=ds.processed_wf,
        signal_begin_distance=ds.ground_peak_dist - 40.0,
        signal_end_distance=ds.ground_peak_dist - 35.0,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_ground = sig_beg_to_ground.transpose(dims[0], dims[1], dims[2])
    ht_35_to_40m = ht_35_to_40m.transpose(dims[0], dims[1], dims[2])

    return ht_35_to_40m.sum(dim="rec_bin") / sig_beg_to_ground.sum(dim="rec_bin")


def proportion_sig_beg_to_ground(ds):
    """
    The total energy from signal beginning to the start of the ground peak,
    normalized by total energy of the waveform. Ground peak assumed to be the last peak.
    """
    ds = get_metric_value(ds, metric='ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    sig_beg_to_ground = select_valid_area(
        bins=ds.rec_wf_sample_distance,
        wf=ds.processed_wf,
        signal_begin_distance=ds.sig_begin_dist,
        signal_end_distance=ds.ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    sig_beg_to_ground = sig_beg_to_ground.transpose(dims[0], dims[1], dims[2])
    # reverse denoise since the total energy is also including noise
    sig_beg_to_ground = sig_beg_to_ground + ds.noise_mean

    # total energy of the raw waveform
    total = ds.rec_wf.sum(dim="rec_bin")

    return sig_beg_to_ground.sum(dim="rec_bin") / total


def pct_canopy_cover(ds, cutoff_height=1.0):
    """
    Mean percent canopy cover (range 0-100).
    Cover was computed by dividing returns above the cover height threshold by total
    number of returns (including those below the ground height cut-off) that were in the plot.
    cutoff_height is in meters, return values are in % (0-100)
    """
    ds = get_metric_value(ds, metric='ground_peak_dist')

    # the processed wf is from sig beg to sig end, select sig beg to ground peak
    vegetation_returns = select_valid_area(
        bins=ds.rec_wf_sample_distance,
        wf=ds.processed_wf,
        signal_begin_distance=ds.ground_peak_dist - cutoff_height,
        signal_end_distance=ds.ground_peak_dist,
    )

    # make sure dimensions matches up
    dims = ds.processed_wf.dims
    vegetation_returns = vegetation_returns.transpose(dims[0], dims[1], dims[2])

    # normalized by the total energy between signal beginning and end
    return vegetation_returns.sum(dim="rec_bin") / ds.processed_wf.sum(dim="rec_bin") * 100


def all_zero_variable(ds, template_var='sig_begin_dist'):
    """
    Returns all zeros in a dataarray in the shape of the template var in terms of record index and shot number
    """
    return xr.DataArray(
        0,
        dims=["record_index", "shot_number"],
        coords=[ds[template_var].coords["record_index"], ds[template_var].coords["shot_number"]],
    )


def terrain_relief_class_1(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_1=elev_diff≥0 m and elev_diff<7 m.
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='end_of_ground_peak_dist'
    )
    return xr.where((relief >= 0 & relief < 7), x=1, y=0)


def terrain_relief_class_2(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_2=elev_diff≥7 m and elev_diff<15 m.
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='end_of_ground_peak_dist'
    )
    return xr.where((relief >= 7 & relief < 15), x=1, y=0)


def terrain_relief_class_3(ds):
    """
    Dummy variables used by Duncanson et al. (2010b). Elevation difference within GLAS footprints:
    elev_diff=elev_max - elev_min (meters). terrain_3=elev_diff≥15 m
    """
    relief = get_heights_from_distance(
        ds, top_metric='start_of_ground_peak_dist', bottom_metric='end_of_ground_peak_dist'
    )
    return xr.where(relief >= 15, x=1, y=0)


def plot_shot(record):
    bins = record.rec_wf_sample_distance
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
    "ground_peak_dist": get_ground_peak_dist,
    "adj_ground_peak_dist": get_adj_ground_peak_dist,  # TODO: implement
    "quadratic_mean_dist": get_quadratic_mean_dist,
    "mean_dist": get_mean_dist,
    "pct_05_dist": get_05th_pct_dist,
    "pct_10_dist": get_10th_pct_dist,
    "pct_20_dist": get_20th_pct_dist,
    "pct_25_dist": get_25th_pct_dist,
    "pct_30_dist": get_30th_pct_dist,
    "pct_50_dist": get_50th_pct_dist,
    "pct_60_dist": get_60th_pct_dist,
    "pct_75_dist": get_75th_pct_dist,
    "pct_80_dist": get_80th_pct_dist,
    "pct_90_dist": get_90th_pct_dist,
    "wf_max_e_dist": get_wf_max_e_dist,
    "start_of_adj_ground_peak_dist": get_start_of_adj_ground_peak_dist,  # TODO: implement
    "start_of_ground_peak_dist": get_start_of_ground_peak_dist,  # TODO: implement
    "end_of_ground_peak_dist": get_end_of_ground_peak_dist,  # TODO: implement
}


def get_metric_value(ds, metric):
    if metric not in ds and metric in DISTANCE_METRICS_MAP:
        ds[metric] = DISTANCE_METRICS_MAP[metric](ds)
    elif metric not in ds:
        raise NotImplementedError(
            f'Metric {metric} not found in dataset and not included in DISTANCE_METRICS_MAP for calculation'
        )

    return ds


def get_heights_from_distance(ds, top_metric, bottom_metric):
    # check if the metric is in input ds, recalculate if not
    for metric in [top_metric, bottom_metric]:
        ds = get_metric_value(ds, metric)

    # multiply with -1 since distance is measured from satellite to object, thus top has a smaller value
    return -1 * (ds[top_metric] - ds[bottom_metric])


HEIGHT_METRICS_MAP = {
    "HEIGHT2": sig_beg_to_ground_ht,
    "ht_adjusted": sig_beg_to_adj_ground_ht,
    "QMCH": quadratic_mean_to_ground_ht,
    "MeanH": mean_to_ground_ht,
    "HOME": median_to_sig_end_ht,
    "HOME_Yavasli": centroid_to_adj_ground_ht,
    "pct_HOME_Yavasli": ratio_centroid_to_max_ht,
    "h25_Neigh": pct_25_to_ground_ht,
    "h50_Neigh": pct_50_to_ground_ht,
    "h75_Neigh": pct_75_to_ground_ht,
    "h90_Neigh": pct_90_to_ground_ht,
    "h05_Nelson": pct_05_to_sig_end_ht,
    "h10_Nelson": pct_10_to_sig_end_ht,
    "h20_Nelson": pct_20_to_sig_end_ht,
    "h25_Nelson": pct_25_to_sig_end_ht,
    "h30_Nelson": pct_30_to_sig_end_ht,
    "h50_Nelson": pct_50_to_sig_end_ht,
    "h75_Nelson": pct_75_to_sig_end_ht,
    "h80_Nelson": pct_80_to_sig_end_ht,
    "h90_Nelson": pct_90_to_sig_end_ht,
    # TODO: confirm whether the Baccini metrics are different in terms of ground reference
    "H10_Baccini": pct_10_to_sig_end_ht,
    "H25_Baccini": pct_25_to_sig_end_ht,
    "H60_Baccini": pct_60_to_sig_end_ht,
    "f_slope": front_slope_to_surface_energy_ratio,  # TODO: implement
    "trail_Nelson": adj_ground_to_sig_end_ht,
    "lead_Nelson": start_to_centroid_adj_ground_ht,
    "wf_max_e": highest_energy_value,
    "wf_variance": wf_variance,
    "wf_skew": wf_skew,
    "startpeak": sig_beg_to_highest_energy_ht,
    "wf_n_gs": number_of_gaussian_peaks,
    "Height_35_to_40": proportion_35_to_40m,
    "CANOPY_DEP": sig_beg_to_start_of_ground_peak_ht,
    "CANOPY_ENE": proportion_sig_beg_to_ground,
    "treecover2000_mean": pct_canopy_cover,  # TODO: double check the other two papers
    "acq3_Neigh": all_zero_variable,
    "acq2_Nelson": all_zero_variable,
    "acq3_Nelson": all_zero_variable,
    "ct2_Nelson": all_zero_variable,
    "terrain_1": terrain_relief_class_1,
    "terrain_2": terrain_relief_class_2,
    "terrain_3": terrain_relief_class_3,
}
