import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def get_mean_distance(bins, wf):
    """
    bins and wf are expected to be data arrays with rec_bin, record_index, shot_number as dims
    """
    return bins.weighted(wf).mean("rec_bin")


def get_percentile_distance(bins, wf, percentiles):
    """
    bins and wf are expected to be data arrays with rec_bin, record_index, shot_number as dims
    percentiles is a list in hundredth format (e.g. to get 10th percentile input value 10)
    """
    total = wf.sum(dim="rec_bin")
    cumsum = wf.cumsum(dim="rec_bin")

    output = {}
    for p in percentiles:
        target = total * p / 100.0
        output[p] = bins.where(cumsum > target).max(dim="rec_bin")

    return output


def get_ground_peak_distance(bins, wf, buffer=1):
    """
    bins and wf are expected to be data arrays with rec_bin, record_index, shot_number as dims
    buffer is the minimum index where the ground peak can be located
    """
    assert buffer >= 1

    # ensure that things are ordered the same way
    bins = bins.transpose("rec_bin", "record_index", "shot_number")
    wf = wf.transpose("rec_bin", "record_index", "shot_number")

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


def get_heights_from_distance(ds, list_of_distance_vars, referece_distance_var):
    for distance_var in list_of_distance_vars:
        output_var = distance_var.replace("distance", "height")
        ds[output_var] = -1 * (ds[distance_var] - ds[referece_distance_var])

    return ds


def plot_shot(record):
    bins = record.rec_wf_sample_distance
    plt.figure(figsize=(6, 10))
    plt.scatter(record.rec_wf, bins, s=5, label="Raw")  # raw wf

    # plot various variables found in GLAH14
    plt.plot(
        [-0.05, 0.5],
        np.array([record.sig_begin_distance, record.sig_begin_distance]),
        "r--",
        label="Signal Beginning",
    )
    plt.plot(
        [-0.05, 0.5],
        np.array([record.sig_end_distance, record.sig_end_distance]),
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
    plt.plot([-0.05, 0.5], [record.meanH_distance, record.meanH_distance], "c--", label="Mean H")
    plt.plot(
        [-0.05, 0.5],
        [record["90th_distance"], record["90th_distance"]],
        "m--",
        label="90th Percentile",
    )
    plt.plot(
        [-0.05, 0.5],
        [record.ground_peak_distance, record.ground_peak_distance],
        "y--",
        label="Ground Peak",
    )

    plt.gca().invert_yaxis()
    plt.xlabel("lidar return (joules)")
    plt.ylabel("distance from satelite (m)")
    plt.legend()
    plt.show()
    plt.close()


HEIGHT_METRICS_MAP = {"meanH": get_mean_distance}
