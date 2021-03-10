import fsspec
import numpy as np
import xarray as xr

from carbonplan_trace.v1.glas_height_metrics import HEIGHT_METRICS_MAP, get_all_height_metrics

ECOREGIONS_GROUPINGS = {
    'afrotropic': np.arange(1, 117),
    'tropical_asia': np.concatenate(
        (
            np.arange(135, 142),
            np.arange(143, 147),
            np.arange(151, 167),
            np.arange(217, 324),
            np.array([148, 149, 188, 195, 618, 621, 622, 626, 627, 634, 635, 637, 638]),
        ),
        axis=None,
    ),
    'tropical_neotropic': np.concatenate(
        (
            np.arange(439, 561),
            np.arange(564, 575),
            np.arange(579, 585),
            np.arange(587, 618),
            np.arange(622, 626),
            # 634 showed up in both tropical asia and here, determined to be more suitable for tropical asia
            np.arange(628, 634),
            np.arange(639, 642),
            np.array([562, 619, 620, 636]),
        ),
        axis=None,
    ),
    'extratropical_neotropic': np.concatenate(
        (
            np.arange(575, 579),
            np.array([561, 563, 585, 586]),
        ),
        axis=None,
    ),
    'alaska': np.concatenate(
        (
            np.arange(404, 412),
            np.array([369, 371, 372, 375, 416, 420]),
        ),
        axis=None,
    ),
    'western_canada': np.concatenate(
        (
            np.arange(376, 382),
            np.arange(412, 416),
            np.array([383, 419]),
        ),
        axis=None,
    ),
    'eastern_canada': np.array([370, 373, 374, 382, 421]),
    'conus': np.concatenate(
        (
            np.arange(328, 369),
            np.arange(384, 404),
            np.arange(422, 426),
            np.array([325, 429, 430, 433, 434, 438]),
        ),
        axis=None,
    ),
    'mexico_north': np.array([324, 326, 327, 426, 428, 431, 432, 435, 436, 437]),
    'mexico_south': np.array([427]),
    'western_boreal_eurasia': np.array([691, 708, 711, 717, 729, 774, 776, 778, 780]),
    'eastern_boreal_eurasia': np.concatenate(
        (
            np.arange(712, 717),
            np.arange(718, 721),
            np.arange(771, 774),
            np.arange(781, 785),
            np.array([710, 775, 777, 779]),
        ),
        axis=None,
    ),
    'palearctic_wang_2013': np.concatenate(
        (
            np.arange(655, 658),
            np.arange(704, 708),
            np.arange(721, 725),
            np.arange(726, 729),
            np.arange(730, 744),
            np.arange(746, 758),
            np.arange(759, 771),
            np.array(
                [
                    642,
                    643,
                    653,
                    659,
                    667,
                    669,
                    673,
                    677,
                    680,
                    681,
                    684,
                    685,
                    687,
                    690,
                    693,
                    694,
                    696,
                    697,
                    700,
                    702,
                    709,
                ]
            ),
        ),
        axis=None,
    ),
    'palearctic_takagi_2015': np.array([666, 670, 671, 682, 683, 698, 699]),
    'palearctic_yavasli_2016': np.array([649, 652, 662, 688, 725, 785, 786, 789, 790, 791, 804]),
    'palearctic_brovkina_2015': np.array([692]),
    'palearctic_alberti_2013': np.array([689]),
    'palearctic_whrc': np.array(
        [644, 646, 650, 658, 660, 665, 674, 675, 678, 695, 703, 788, 794, 795, 799, 801, 802, 806]
    ),
    'palearctic_shang_and_chazette_2014': np.array(
        [645, 647, 648, 654, 661, 664, 668, 676, 679, 686]
    ),
    'palearctic_simonson_2016': np.array([701, 758, 787, 792, 793, 796, 797, 798, 800, 803, 805]),
    'palearctic_patenaude_2004': np.array([651, 663, 672]),
    'palearctic_suganuma_2006': np.concatenate(
        (
            np.arange(807, 847),
            np.array([744, 745]),
        ),
        axis=None,
    ),
    'australia_beets_2011': np.concatenate(
        (
            np.arange(169, 176),
            np.array([142, 147, 180, 190, 194, 196]),
        ),
        axis=None,
    ),
    'australia_suganuma_2006': np.arange(197, 217),
    'australia_lucas_2008': np.concatenate(
        (
            np.arange(176, 180),
            np.arange(181, 188),
            np.arange(191, 194),
            np.array([167, 168, 189]),
        ),
        axis=None,
    ),
    'australia_baccini_2012': np.array([150]),
}


CONUS_CONIFER_GROUPING = {
    'conus_conifer_nelson_2017': np.concatenate(
        (
            np.arange(387, 393),
            np.arange(394, 399),
            np.arange(400, 403),
            np.arange(422, 426),
            np.array([325, 356, 385, 429, 430, 433, 434, 438]),
        ),
        axis=None,
    ),
    'conus_conifer_tsui_2012': np.array([349, 351, 360, 364, 365, 403]),
    'conus_conifer_neigh_2013': np.array([345, 350, 355, 361, 362, 386]),
    'conus_conifer_lu_2012': np.array([352, 354, 357, 358, 359, 366]),
    'conus_conifer_hudak_2012': np.array([348, 353, 367, 368]),
    'conus_conifer_hyde_2007': np.array([346]),
    'conus_conifer_popescu_2011': np.array(
        [329, 330, 331, 332, 336, 337, 340, 341, 363, 384, 393, 399]
    ),
    'conus_conifer_skowronski_2007': np.array([347]),
    'conus_conifer_sun_2011': np.array([333, 334, 335, 338]),
    'conus_conifer_anderson_2006': np.array([328, 339, 342, 343, 344]),
}


def tropics(ds):
    required_metrics = ['HOME_Baccini', 'H10_Baccini', 'H25_Baccini', 'H60_Baccini', 'CANOPY_ENE']
    ds = get_all_height_metrics(ds, required_metrics)
    return (
        -31.631
        + 15.952 * ds['HOME_Baccini']
        + 7.832 * ds['H10_Baccini']
        - 18.805 * ds['H60_Baccini']
        - 38.428 * ds['CANOPY_ENE']
        + 8.285 * ds['H25_Baccini']
    )


def extratropical_neotropic(ds):
    required_metrics = ['treecover2000_mean']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (0.41 + 0.53 * ds['treecover2000_mean'])


def ak_wetland_steep(ds):
    required_metrics = ['VH', 'h50_Neigh', 'f_slope', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return (
        4.48 * ds['VH']
        - 6.27 * ds['h50_Neigh']
        - 2.10 * ds['f_slope']
        + 6.31 * ds['acq3_Neigh']
        + 11.66
    )


def boreal_wetland(ds):
    required_metrics = ['VH', 'h50_Neigh', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.52 * ds['VH'] - 5.17 * ds['h50_Neigh'] - 2.22 * ds['acq3_Neigh'] + 4.82


def ak_hardwood(ds):
    required_metrics = ['h90_Neigh', 'h25_Neigh', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.28 * ds['h90_Neigh'] - 4.34 * ds['h25_Neigh'] + 15.95 * ds['acq3_Neigh'] + 37.16


def ak_conifer(ds):
    required_metrics = ['h90_Neigh', 'h50_Neigh', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 7.39 * ds['h90_Neigh'] - 8.75 * ds['h50_Neigh'] - 1.35 * ds['acq3_Neigh'] + 36.98


def ak_mixedwood(ds):
    required_metrics = ['QMCH', 'VH', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 4.54 * ds['QMCH'] + 2.45 * ds['VH'] + 18.96 * ds['acq3_Neigh'] + 5.76


def w_canada_hardwood(ds):
    required_metrics = ['h90_Neigh', 'h25_Neigh', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.28 * ds['h90_Neigh'] - 4.34 * ds['h25_Neigh'] + 15.95 * ds['acq3_Neigh'] + 37.16


def w_canada_conifer(ds):
    required_metrics = ['VH', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.31 * ds['VH'] - 4.19 * ds['h25_Neigh'] + 4.03


def canada_mixedwood(ds):
    required_metrics = ['QMCH', 'VH', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 4.11 * ds['QMCH'] + 2.26 * ds['VH'] - 2.50 * ds['h25_Neigh'] + 20.61


def e_canada_wetland_steep(ds):
    required_metrics = ['VH', 'f_slope']
    ds = get_all_height_metrics(ds, required_metrics)
    return 6.08 * ds['VH'] - 1.29 * ds['f_slope'] - 1.20


def e_canada_hardwood(ds):
    required_metrics = ['QMCH', 'h90_Neigh', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 3.92 * ds['QMCH'] + 6.09 * ds['h90_Neigh'] - 5.44 * ds['h25_Neigh'] + 9.32


def e_canada_conifer(ds):
    required_metrics = ['VH', 'h90_Neigh', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2.27 * ds['VH'] + 5.54 * ds['h90_Neigh'] - 9.13 * ds['h25_Neigh'] + 2.40


def e_canada_mixedwood_high(ds):
    required_metrics = ['VH', 'senergy', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.74 * ds['VH'] - 0.92 * ds['senergy'] - 5.01 * ds['h25_Neigh'] + 14.63


def conus_wetland(ds):
    required_metrics = ['h50_Nelson', 'h90_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    return 3.623 * ds['h50_Nelson'] + 4.980 * ds['h90_Nelson']


def conus_hardwood(ds):
    required_metrics = ['h75_Nelson', 'h10_Nelson', 'trail_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    return 6.660 * ds['h75_Nelson'] - 7.036 * ds['h10_Nelson'] + 3.132 * ds['trail_Nelson']


def conus_mixedwood(ds):
    required_metrics = ['h75_Nelson', 'h10_Nelson', 'lead_Nelson', 'trail_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    return (
        6.147 * ds['h75_Nelson']
        - 6.292 * ds['h10_Nelson']
        + 0.667 * ds['lead_Nelson']
        + 3.128 * ds['trail_Nelson']
    )


def zero_biomass(ds):
    return 0


def conus_conifer_tsui_2012(ds):
    required_metrics = ['MeanH', 'h10_p12', 'h90_p12']
    ds = get_all_height_metrics(ds, required_metrics)
    return -7.144 - 12.925 * ds["MeanH"] + 2.239 * ds["h10_p12"] + 14.019 * ds["h90_p12"]


def conus_conifer_neigh_2013(ds):
    required_metrics = ['VH', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.31 * ds['VH'] - 4.19 * ds['h25_Neigh'] + 4.03


def conus_conifer_lu_2012(ds):
    required_metrics = ['QMCH', 'Height_35_to_40']
    ds = get_all_height_metrics(ds, required_metrics)
    return np.exp((2.10 + 1.56 * np.log(ds['QMCH'])) + (0.05 + np.log(ds['Height_35_to_40'])))


def conus_conifer_hudak_2012(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return -16.54 + 21.18 * ds['MeanH']


def conus_conifer_hyde_2007(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 7.042 + 16.141 * ds['MeanH']


def conus_conifer_popescu_2011(ds):
    required_metrics = ['VH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 7.5429 * ds['VH'] - 29.308


def conus_conifer_skowronski_2007(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 6.04 * ds['MeanH']


def conus_conifer_sun_2011(ds):
    required_metrics = ['h50_Neigh', 'h75_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return -1.717 + 6.208 * ds['h50_Neigh'] + 8.625 * ds['h75_Neigh']


def conus_conifer_anderson_2006(ds):
    required_metrics = ['h50_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 29.954 + 14.297 * ds['h50_Neigh']


ALLOMETRIC_EQUATIONS_MAP = {
    'afrotropic': tropics,
    'tropical_asia': tropics,
    'tropical_neotropic': tropics,
    'extratropical_neotropic': extratropical_neotropic,
    'ak_wetland_steep': ak_wetland_steep,
    'ak_wetland_flat': boreal_wetland,
    'ak_hardwood': ak_hardwood,
    'ak_conifer': ak_conifer,
    'ak_mixedwood': ak_mixedwood,
    'w_canada_wetland': boreal_wetland,
    'w_canada_hardwood': w_canada_hardwood,
    'w_canada_conifer': w_canada_conifer,
    'w_canada_mixedwood': canada_mixedwood,
    'e_canada_wetland_steep': e_canada_wetland_steep,
    'e_canada_wetland_flat': boreal_wetland,
    'e_canada_hardwood': e_canada_hardwood,
    'e_canada_conifer': e_canada_conifer,
    'e_canada_mixedwood_high': e_canada_mixedwood_high,
    'e_canada_mixedwood_low': canada_mixedwood,
    'conus_wetland': conus_wetland,
    'conus_hardwood': conus_hardwood,
    'conus_mixedwood': conus_mixedwood,
    'zero_biomass': zero_biomass,
    # 'mexico_north': mexico_north,
    # 'mexico_south': mexico_south,
    'conus_conifer_nelson_2017': conus_mixedwood,
    'conus_conifer_tsui_2012': conus_conifer_tsui_2012,
    'conus_conifer_neigh_2013': conus_conifer_neigh_2013,
    'conus_conifer_lu_2012': conus_conifer_lu_2012,
    'conus_conifer_hudak_2012': conus_conifer_hudak_2012,
    'conus_conifer_hyde_2007': conus_conifer_hyde_2007,
    'conus_conifer_popescu_2011': conus_conifer_popescu_2011,
    'conus_conifer_skowronski_2007': conus_conifer_skowronski_2007,
    'conus_conifer_sun_2011': conus_conifer_sun_2011,
    'conus_conifer_anderson_2006': conus_conifer_anderson_2006,
    # 'eastern_boreal_eurasia': eastern_boreal_eurasia,
    # 'palearctic_wang_2013': palearctic_wang_2013,
    # 'palearctic_takagi_2015': palearctic_takagi_2015,
    # 'palearctic_yavasli_2016': palearctic_yavasli_2016,
    # 'palearctic_brovkina_2015': palearctic_brovkina_2015,
    # 'palearctic_alberti_2013': palearctic_alberti_2013,
    # 'palearctic_whrc': palearctic_whrc,
    # 'palearctic_shang_and_chazette_2014': palearctic_shang_and_chazette_2014,
    # 'palearctic_simonson_2016': palearctic_simonson_2016,
    # 'palearctic_patenaude_2004': palearctic_patenaude_2004,
    # 'palearctic_suganuma_2006': palearctic_suganuma_2006,
    # 'australia_beets_2011': australia_beets_2011,
    # 'australia_suganuma_2006': australia_suganuma_2006,
    # 'australia_lucas_2008': australia_lucas_2008,
    # 'australia_baccini_2012': australia_baccini_2012
}


def get_list_of_mask_tiles():
    """
    Ecoregions mask is stored in 10 degree tiles, grab the
    """
    fs = fsspec.get_filesystem_class('gs')(account_name='carbonplan')

    folder = 'gs://carbonplan-scratch/trace_scratch/ecoregions_mask/'
    # fs.ls includes the parent folder itself, skip that link
    paths = [tp for tp in fs.ls(folder) if not tp.endswith('/')]

    return paths


def parse_bounding_lat_lon_for_tile(path):
    # grab the file name without folder and extension
    fn = path.split('/')[-1].split('.')[0]
    lat = fn.split('_')[0]
    lon = fn.split('_')[1]

    # the tile name denotes the upper left corner of each tile
    if lat.endswith('N'):
        max_lat = float(lat[:-1])
    elif lat.endswith('S'):
        max_lat = -1 * float(lat[:-1])
    # each tile covers 10 degree x 10 degree
    min_lat = max_lat - 10

    if lon.endswith('E'):
        min_lon = float(lon[:-1])
    elif lon.endswith('W'):
        min_lon = -1 * float(lon[:-1])
    max_lon = min_lon + 10

    return min_lat, max_lat, min_lon, max_lon


def get_ecoregions_mask(path):
    if not path.startswith('gs://'):
        path = 'gs://' + path
    mapper = fsspec.get_mapper(path)
    return xr.open_zarr(mapper)


def get_igbp_data():
    igbp = xr.open_rasterio(
        'gs://carbonplan-scratch/trace_scratch/IGBP.tif', parse_coordinates=True
    )
    igbp = igbp.squeeze(dim='band', drop=True)
    igbp = igbp.rename(x='lon', y='lat')

    return igbp


def subset_data_for_tile(data, tile_path):
    """
    Return a subset of data within the bounding lat/lon box of target tile
    The function assumes that lat/lon are not coordinates in the data
    """
    min_lat, max_lat, min_lon, max_lon = parse_bounding_lat_lon_for_tile(tile_path)
    sub = data.where(
        (data.lat > min_lat) & (data.lat <= max_lat) & (data.lon > min_lon) & (data.lon <= max_lon),
        drop=True,
    )
    return sub


def find_ecoregions_values(data, ecoregions):
    # ecoregions map contains Ecoregions2017 and NLCD data, and will eventually include EOSD
    eco_records = ecoregions.sel(lat=data.lat, lon=data.lon, method="nearest")
    for v in eco_records:
        data[v] = eco_records[v]

    # TODO: remove once we have valid EOSD data
    data['eosd'] = xr.DataArray(
        np.nan,
        dims=["unique_index"],
        coords=[data['lat'].coords["unique_index"]],
    )

    # 0 and 241 are both null values for nlcd
    data['nlcd'] = xr.where(data.nlcd.isin([0, 241]), x=np.nan, y=data.nlcd)

    # TODO: figure out what null values are for EOSD

    # read IGBP data and assign those values to each record as well
    igbp = get_igbp_data()
    igbp_records = igbp.sel(lat=data.lat, lon=data.lon, method="nearest")
    data['igbp'] = igbp_records['igbp']

    return data


def assign_ecoregion_values(ds):
    """
    Given a dataset with lat, lon, assign the correct ecoregions values for each record
    """
    # error out if lat lon are not in dataset
    for v in ['lat', 'lon']:
        if v not in ds:
            raise KeyError(f'required variable {v} not found in input data')

    # get list of ecoregions mask
    tile_paths = get_list_of_mask_tiles()

    output = []
    print(f'Original number of records is {ds.dims["unique_index"]}')
    for path in tile_paths:
        # subset data to lat/lon within this tile, we only need 4 columns from the input
        sub = subset_data_for_tile(data=ds[['lat', 'lon', 'f_slope', 'senergy']], tile_path=path)
        # if there's no data in this tile, skip
        if sub.dims['unique_index'] == 0:
            continue

        print(
            f'    Assigning allometric equations to records within {path.split("/")[-1].split(".")[0]}'
        )
        ecoregions_mask = get_ecoregions_mask(path)

        # find the closest matching cell for each record
        sub = find_ecoregions_values(sub, ecoregions_mask)
        output.append(sub)

    output = xr.concat(output, dims=['unique_index'])
    print(
        f'After matching to tiles, remaining number of record is is {output.dims["unique_index"]}'
    )

    return output


def assign_allometric_eq(ds):
    """
    Given a dataset with ecoregion values (Ecoregions2017, NLCD, IGBP), f_slope, and senergy, assign the correct allometric equation
    """
    # calculate f_slope and senergy if not in dataset
    for v in ['f_slope', 'senergy']:
        if v not in ds:
            print(f'metric {v} not found in input, calculating')
            ds[v] = HEIGHT_METRICS_MAP[v](ds)

    output = xr.DataArray(
        '',
        dims=["unique_index"],
        coords=[ds['lat'].coords["unique_index"]],
    )

    # get the groupings by ecoregions2017
    for name, id_list in ECOREGIONS_GROUPINGS.items():
        output = xr.where(ds.ecoregion.isin(id_list), x=name, y=output)

    # get land cover groupings
    wetland_mask = (
        ds.nlcd == 90 | ds.eosd == 81 | (ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp == 11)
    )
    hardwood_mask = ds.nlcd == 41 | ds.eosd.isin([220, 221, 222, 223]) | (
        ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp == 4
    )
    conifer_mask = ds.nlcd == 42 | ds.eosd.isin([210, 211, 212, 213]) | (
        ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([1, 3])
    )
    mixedwood_mask = ds.nlcd == 43 | ds.eosd.isin([51, 230, 231, 232, 233]) | (
        ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([5, 8, 9])
    )
    # TODO: add burned
    zero_biomass_mask = (
        ds.nlcd.isin([11, 12, 21, 22, 23, 24, 31, 52, 71, 72, 73, 74, 81, 82, 95])
        | ds.eosd.isin([0, 10, 11, 12, 20, 30, 31, 32, 33, 34, 35, 36, 37, 40, 51, 52, 82, 83, 100])
        | (ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([0, 6, 7, 10, 12, 13, 15, 16]))
    )

    # process boreal nearctic
    alaska_mask = output == 'alaska'
    output = xr.where(
        alaska_mask & wetland_mask & ds.f_slope >= 2.552, x='ak_wetland_steep', y=output
    )
    output = xr.where(
        alaska_mask & wetland_mask & ds.f_slope < 2.552, x='ak_wetland_flat', y=output
    )
    output = xr.where(alaska_mask & hardwood_mask, x='ak_hardwood', y=output)
    output = xr.where(alaska_mask & conifer_mask, x='ak_conifer', y=output)
    output = xr.where(alaska_mask & mixedwood_mask, x='ak_mixedwood', y=output)
    output = xr.where(alaska_mask & zero_biomass_mask, x='zero_biomass', y=output)

    western_canada_mask = output == 'western_canada'
    output = xr.where(western_canada_mask & wetland_mask, x='w_canada_wetland', y=output)
    output = xr.where(western_canada_mask & hardwood_mask, x='w_canada_hardwood', y=output)
    output = xr.where(western_canada_mask & conifer_mask, x='w_canada_conifer', y=output)
    output = xr.where(western_canada_mask & mixedwood_mask, x='w_canada_mixedwood', y=output)
    output = xr.where(western_canada_mask & zero_biomass_mask, x='zero_biomass', y=output)

    eastern_canada_mask = output == 'eastern_canada'
    output = xr.where(
        eastern_canada_mask & wetland_mask & ds.f_slope >= 2.552,
        x='e_canada_wetland_steep',
        y=output,
    )
    output = xr.where(
        eastern_canada_mask & wetland_mask & ds.f_slope < 2.552, x='e_canada_wetland_flat', y=output
    )
    output = xr.where(eastern_canada_mask & hardwood_mask, x='e_canada_hardwood', y=output)
    output = xr.where(eastern_canada_mask & conifer_mask, x='e_canada_conifer', y=output)
    output = xr.where(
        eastern_canada_mask & mixedwood_mask & ds.senergy >= 26.643,
        x='e_canada_mixedwood_high',
        y=output,
    )
    output = xr.where(
        eastern_canada_mask & mixedwood_mask & ds.senergy < 26.643,
        x='e_canada_mixedwood_low',
        y=output,
    )
    output = xr.where(eastern_canada_mask & zero_biomass_mask, x='zero_biomass', y=output)

    # process conus ecoregions
    conus_mask = output == 'conus'
    output = xr.where(conus_mask & wetland_mask, x='conus_wetland', y=output)
    output = xr.where(conus_mask & hardwood_mask, x='conus_hardwood', y=output)
    output = xr.where(conus_mask & mixedwood_mask, x='conus_mixedwood', y=output)
    output = xr.where(conus_mask & zero_biomass_mask, x='zero_biomass', y=output)

    # process conus conifer
    for name, id_list in CONUS_CONIFER_GROUPING.items():
        output = xr.where(ds.ecoregion.isin(id_list) & conifer_mask, x=name, y=output)

    # process western boreal eurasia
    wbe_mask = output == 'western_boreal_eurasia'
    conifer_or_woody_mask = ds.igbp.isin([1, 3, 11, 8, 9])
    output = xr.where(wbe_mask & conifer_or_woody_mask, x='wbe_confier', y=output)
    mixed_and_hard_wood_mask = ds.igbp.isin([0, 2, 4, 5, 6, 7, 10, 12, 13, 14, 15, 16])
    output = xr.where(wbe_mask & mixed_and_hard_wood_mask, x='wbe_mixed_hard_wood', y=output)

    return output


def apply_allometric_equation(ds):
    """
    Given a dataset containing GLAS records, assign allometric equation based on appropriate info,
    then apply the allometric equation with respective height metrics to get biomass
    """
    # assign ecoregions
    ds = assign_ecoregion_values(ds)

    # assign allometric eq
    ds["allometric_eq"] = assign_allometric_eq(ds)

    # # apply allometric equations
    # ds["biomass"] = tsui_etal_2012(ds)

    return 0
