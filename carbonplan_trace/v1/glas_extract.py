import h5py
import numpy as np
import pandas as pd
import xarray as xr

NAN_THRESHOLD = 1e30


def read_dimensions(file_handle):
    # put the dimension columns into a dataframe
    df = {}
    df["record_index"] = file_handle["Data_40HZ/Time/i_rec_ndx"][:]
    df["shot_number"] = file_handle["Data_40HZ/Time/i_shot_count"][:]
    df = pd.DataFrame(df)

    # concat two columns to get an unique index, but also set these two columns as multi level index
    df["unique_index"] = (
        df.record_index.astype(str).str.zfill(9) + "_" + df.shot_number.astype(str).str.zfill(2)
    )
    df.set_index(["record_index", "shot_number"], inplace=True)

    return df


def read_1d_variables(file_handle, mapping, unique_index, replace_fill_values_with_nulls):
    # put the 1D variables into a xarray dataset
    ds = {}
    for k, v in mapping.items():
        temp = file_handle[v][:]
        if replace_fill_values_with_nulls and (temp > NAN_THRESHOLD).sum() > 0:
            print(
                f'Found {(temp > NAN_THRESHOLD).sum()} values greater than threshold of {NAN_THRESHOLD} in variable {k}, replacing with np.nan'
            )
            temp = temp * 1.0
            temp[(temp > NAN_THRESHOLD)] = np.nan

        ds[k] = xr.DataArray(temp, dims=["unique_index"], coords={"unique_index": unique_index})

    return xr.Dataset(ds)


def extract_GLAH01_data(filename, replace_fill_values_with_nulls=True):
    """
    Given file name of a HDF5 GLAH01 data, returns a xarray dataset with record index and shot number being the primary dimensions.
    rec_bin and tx_bin are additional dimensions regarding the transmitted and received waveforms.
    """

    # read file
    f = h5py.File(filename, "r")

    # list out all 1D variables we want and read into a dataframe
    name_map = {
        # Background Noise Mean Value for the 4 ns filter. From APID12/13, Offset 112.
        "noise_mean": "Data_40HZ/Waveform/Characteristics/d_4nsBgMean",  # volts
        # The standard deviation of the background noise for the 4 ns filter. From APID12/13, Offset 116.
        "noise_sd": "Data_40HZ/Waveform/Characteristics/d_4nsBgSDEV",  # volts
        "rec_wf_location_ind": "Data_40HZ/Waveform/RecWaveform/i_rec_wf_location_index",  # This is an index into the array of 544 times within the rec_wf_sample_location_table (found in the ANCILLARY_DATA group)
        "rec_wf_response_end_time": "Data_40HZ/Waveform/RecWaveform/i_RespEndTime",
        "tx_wf_peak_time": "Data_40HZ/Waveform/TransmitWaveform/i_time_txWfPk",  # Address in digitizer counts of the Transmit Pulse Peak as measured from the start of Acquisition Memory, i.e. start of digitization. From APID12/13, Offset 68.
        #     'wf_type': 'Data_40HZ/Waveform/Characteristics/i_waveformType', # Indicates number of valid samples in waveform; 0 = missing; 1 = Long waveform (544 samples); 2 =Short waveform (200 samples),
        #     'tx_wf_start_time': 'Data_40HZ/Waveform/TransmitWaveform/i_TxWfStart'  # Starting Address in digitizer counts of the Transmit Pulse sample relative to the start of digitization. From APID12/13, Offset 76.
    }

    # put the dimension columns into a dataframe
    df = read_dimensions(f)

    # put the 1D variables into a xarray dataset
    ds = read_1d_variables(
        file_handle=f,
        mapping=name_map,
        unique_index=df.unique_index.values,
        replace_fill_values_with_nulls=replace_fill_values_with_nulls,
    )

    # read the 2D variables we want
    # Transmit Pulse 48 waveform samples in calibrated volts. The delta times for transmit waveform sample j is provided in the attribute array tx_wf_sample_location_table (j).
    tx_wf = f["Data_40HZ/Waveform/TransmitWaveform/r_tx_wf"][:]
    # The delta times for each echo of the 544 waveform samples is provided within the 544 times stored in rec_wf_sample_location_table
    # (an attribute in the /ANCILLARY_DATA group) and indexed by i_rec_wf_location_index.
    rec_wf = f["Data_40HZ/Waveform/RecWaveform/r_rng_wf"][:]  # n (num shot * num records) x 544
    tx_wf_sample_loc = f["ANCILLARY_DATA"].attrs["tx_wf_sample_location_table"]
    rec_wf_sample_loc = f["ANCILLARY_DATA"].attrs["rec_wf_sample_location_table"]  # 5 x 544

    if replace_fill_values_with_nulls:
        rec_wf_sample_loc[(rec_wf_sample_loc > 1e10)] = np.nan

    # put the 2D variables into xarray
    ds["rec_wf"] = xr.DataArray(
        rec_wf,
        dims=["unique_index", "rec_bin"],
        coords=[df.unique_index.values, np.arange(rec_wf.shape[1])],
    )
    ds["tx_wf"] = xr.DataArray(
        tx_wf,
        dims=["unique_index", "tx_bin"],
        coords=[df.unique_index.values, np.arange(tx_wf.shape[1])],
    )
    ds["tx_wf_sample_loc"] = xr.DataArray(
        tx_wf_sample_loc, dims=["tx_bin"], coords=[np.arange(tx_wf.shape[1])]
    )

    # store a copy of the sample location for each unique shot
    # value 127 is the largest for int8 and represents a null in this field
    null_loc = ds.rec_wf_location_ind == 127
    ind = xr.where(null_loc, 0, ds.rec_wf_location_ind - 1)
    loc = rec_wf_sample_loc[ind]
    loc[null_loc] = np.nan
    ds["rec_wf_sample_loc"] = xr.DataArray(
        loc,
        dims=["unique_index", "rec_bin"],
        coords=[df.unique_index.values, np.arange(rec_wf.shape[1])],
    )

    # expand the multi index
    ds.coords["unique_index"] = df.index
    ds = ds.unstack("unique_index")

    return ds


def extract_GLAH14_data(filename, replace_fill_values_with_nulls=True):
    """
    Given file name of a HDF5 GLAH14 data, returns a xarray dataset with record index and shot number being the primary dimensions.
    rec_bin and tx_bin are additional dimensions regarding the transmitted and received waveforms.
    """

    # read file
    f = h5py.File(filename, "r")

    # list out all 1D variables we want and read into a dataframe
    name_map = {
        # The transmit time of each shot in the 1 second frame measured as UTC seconds elapsed since Jan 1 2000 12:00:00 UTC. This time has been derived from the GPS time accounting for leap seconds.
        "time": "Data_40HZ/Time/d_UTCTime_40",
        "lat": "Data_40HZ/Geolocation/d_lat",
        "lon": "Data_40HZ/Geolocation/d_lon",
        # the documentation mentioned two flags sat_corr_flg and i_satNdx to signal bad elevation, also and when correction is invalid the elevation is invalid
        "elevation": "Data_40HZ/Elevation_Surfaces/d_elev",  # meters
        "elevation_correction": "Data_40HZ/Elevation_Corrections/d_satElevCorr",  # should be added to elevation
        "elevation_SRTM": "Data_40HZ/Geophysical/d_DEM_elv",  # Elevation at the footprint location from the SRTM30 (GTOPO30 + SRTM) Digital Elevation Model (DEM).
        "delta_ellipse": "Data_40HZ/Geophysical/d_deltaEllip",
        "geoid": "Data_40HZ/Geophysical/d_gdHt",
        # Range in distance calculated from the time between the centroid of the transmit pulse and the farthest gate from the spacecraft of the received pulse. See the rngcorrflg to determine
        # any corrections that have been applied. unit is meters and values in the 600k range
        "ref_range": "Data_40HZ/Elevation_Surfaces/d_refRng",  # meters
        # these should be added to centroid according to the documentation
        "sig_begin_offset": "Data_40HZ/Elevation_Offsets/d_SigBegOff",  # meters
        "sig_end_offset": "Data_40HZ/Elevation_Offsets/d_SigEndOff",  # meters
        # Range offset to be added to d_refRng to calculate the range using the algorithm deemed appropriate for land.
        "centroid_offset": "Data_40HZ/Elevation_Offsets/d_ldRngOff",  # meters
        # data for the 6 fitted gaussian peaks
        'num_gaussian_peaks': 'Data_40HZ/Waveform/i_numPk',
        # other waveform metric
        'skew': 'Data_40HZ/Waveform/d_skew1',
        # The initial number of peaks of the received echo; determined from the smoothed waveform, using alternate parameters
        'n_peaks': 'Data_40HZ/Waveform/i_nPeaks1',
    }

    # put the dimension columns into a dataframe
    df = read_dimensions(f)

    # put the 1D variables into a xarray dataset
    ds = read_1d_variables(
        file_handle=f,
        mapping=name_map,
        unique_index=df.unique_index.values,
        replace_fill_values_with_nulls=replace_fill_values_with_nulls,
    )

    # read the params of fitted gaussian peaks (6-element arrays)
    gaussian_fit_params = {
        'gaussian_mu': 'Data_40HZ/Elevation_Offsets/d_gpCntRngOff',  # meters
        'gaussian_amp': 'Data_40HZ/Waveform/d_Gamp',  # volts
        'gaussian_sigma': 'Data_40HZ/Waveform/d_Gsigma',  # ns
    }

    for k, v in gaussian_fit_params.items():
        temp = f[v][:]
        if replace_fill_values_with_nulls and (temp > NAN_THRESHOLD).sum() > 0:
            # print(
            #     f'Found {(temp > NAN_THRESHOLD).sum()} values greater than threshold of {NAN_THRESHOLD} in variable {k}, replacing with np.nan'
            # )
            temp.dtype = float
            temp[(temp > NAN_THRESHOLD)] = np.nan

        ds[k] = xr.DataArray(
            temp,
            dims=["unique_index", "n_gaussian_peaks"],
            coords=[df.unique_index.values, np.arange(6)],
        )

    # expand the multi index
    ds.coords["unique_index"] = df.index
    ds = ds.unstack("unique_index")

    return ds
