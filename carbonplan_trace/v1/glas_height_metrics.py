import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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


def get_adj_ground_peak_dist(ds):
    # compare the amplitude of the last two peaks and mark the greater one
    # get the corresponding distance of the gaussian fit
    # return this distance
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


def front_slope_to_surface_energy_ratio(ds):
    """
    Front slope to surface energy ratio. We calculated fslope_WHRC as the change in amplitude per meter (volts/meter) in the outer canopy.
    We then applied the following linear transformation in order to calculate fslope on the same scale as provided in data published by
    Margolis et al. (2015): f_slope  =0.5744 + 19.7762⋅fslope_WHRC
    """
    # find canopy_peak dist = distance of highest Gaussian peak
    # find canopy_peak amp = amplitude of highest Gaussian peak - signal noise  --> may not work and need to find the first peak manually
    # TODO: is this the first peak or the biggest peak...???
    # find sig beg dist
    # find amplitude of the sig beg = 0
    # calculate slope as y2-y1 / x2-x1
    fslope_WHRC = 0
    return 0.5744 + 19.7762 * fslope_WHRC


def width_of_adj_ground_pk(ds):
    """
    Distance, in meters, from the start of the ground signal to the ground peak.
    Ground peak defined as whichever of the two lowest peaks has greater amplitude.

    vertical distance, start of the Gaussian ground signal to the ground peak (m)
    """
    # find the adjusted ground peak
    # find the distance at which gaussian signal would be lower than noise based on amp, sigma, and loation
    # get diff between the acutal peak and the signal start

    pass


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
}


def get_heights_from_distance(ds, top_metric, bottom_metric):
    # check if the metric is in input ds, recalculate if not
    for metric in [top_metric, bottom_metric]:
        if metric not in ds and metric in DISTANCE_METRICS_MAP:
            ds[top_metric] = DISTANCE_METRICS_MAP[metric](ds)
        elif metric not in ds:
            raise NotImplementedError(
                f'Metric {metric} not found in dataset and not included in DISTANCE_METRICS_MAP for calculation'
            )

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
    "lead_Nelson": width_of_adj_ground_pk,  # TODO: implement
}


# TODO:
# get centroid cntRngOff
# get gaussian fits
