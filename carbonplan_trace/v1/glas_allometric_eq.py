from datetime import datetime, timezone

import numpy as np
import xarray as xr

from carbonplan_trace.v0.data import cat
from carbonplan_trace.v1 import utils
from carbonplan_trace.v1.glas_height_metrics import HEIGHT_METRICS_MAP, get_all_height_metrics
from carbonplan_trace.v1.glas_preprocess import get_modeled_waveform

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
            np.arange(417, 420),
            np.array([383]),
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


REALM_GROUPINGS = {
    'afrotropic': ECOREGIONS_GROUPINGS['afrotropic'],
    'australia': np.concatenate(
        (
            ECOREGIONS_GROUPINGS['australia_beets_2011'],
            ECOREGIONS_GROUPINGS['australia_suganuma_2006'],
            ECOREGIONS_GROUPINGS['australia_lucas_2008'],
            ECOREGIONS_GROUPINGS['australia_baccini_2012'],
        ),
        axis=None,
    ),
    'nearctic': np.concatenate(
        (
            ECOREGIONS_GROUPINGS['alaska'],
            ECOREGIONS_GROUPINGS['western_canada'],
            ECOREGIONS_GROUPINGS['eastern_canada'],
            ECOREGIONS_GROUPINGS['conus'],
            ECOREGIONS_GROUPINGS['mexico_north'],
            ECOREGIONS_GROUPINGS['mexico_south'],
        ),
        axis=None,
    ),
    'neotropic': np.concatenate(
        (
            ECOREGIONS_GROUPINGS['tropical_neotropic'],
            ECOREGIONS_GROUPINGS['extratropical_neotropic'],
        ),
        axis=None,
    ),
    'palearctic': np.concatenate(
        (
            ECOREGIONS_GROUPINGS['western_boreal_eurasia'],
            ECOREGIONS_GROUPINGS['eastern_boreal_eurasia'],
            ECOREGIONS_GROUPINGS['palearctic_wang_2013'],
            ECOREGIONS_GROUPINGS['palearctic_takagi_2015'],
            ECOREGIONS_GROUPINGS['palearctic_yavasli_2016'],
            ECOREGIONS_GROUPINGS['palearctic_brovkina_2015'],
            ECOREGIONS_GROUPINGS['palearctic_alberti_2013'],
            ECOREGIONS_GROUPINGS['palearctic_whrc'],
            ECOREGIONS_GROUPINGS['palearctic_shang_and_chazette_2014'],
            ECOREGIONS_GROUPINGS['palearctic_simonson_2016'],
            ECOREGIONS_GROUPINGS['palearctic_patenaude_2004'],
            ECOREGIONS_GROUPINGS['palearctic_suganuma_2006'],
        ),
        axis=None,
    ),
    'tropical_asia': ECOREGIONS_GROUPINGS['tropical_asia'],
}


ECO_TO_REALM_MAP = {}
for realm, list_of_eco in REALM_GROUPINGS.items():
    new_map = {eco: realm for eco in list_of_eco}
    ECO_TO_REALM_MAP.update(new_map)
for i in [0, 129, 133]:
    ECO_TO_REALM_MAP[i] = 'ice'


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
    if 'modeled_wf' not in ds:
        ds['modeled_wf'] = get_modeled_waveform(ds)
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


def w_canada_hardwood(ds):
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


def w_canada_conifer(ds):
    required_metrics = ['VH', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.31 * ds['VH'] - 4.19 * ds['h25_Neigh'] + 4.03


def canada_mixedwood(ds):
    required_metrics = ['QMCH', 'VH', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 4.11 * ds['QMCH'] + 2.26 * ds['VH'] - 2.50 * ds['h25_Neigh'] + 20.61


def w_canada_burned(ds):
    required_metrics = ['QMCH', 'ht_adjusted', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.46 * ds['QMCH'] + 4.07 * ds['ht_adjusted'] + 22.77 * ds['acq3_Neigh'] + 5.05


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


def e_canada_burned(ds):
    required_metrics = ['QMCH', 'ht_adjusted', 'acq3_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.46 * ds['QMCH'] + 4.07 * ds['ht_adjusted'] + 22.76 * ds['acq3_Neigh'] + 5.05


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


def conus_burned(ds):
    required_metrics = ['h90_Nelson', 'h75_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    return 4.331 * ds['h90_Nelson'] - 4.222 * ds['h75_Nelson']


def zero_biomass(ds):
    return xr.DataArray(
        0,
        dims=["unique_index"],
        coords=[ds['lat'].coords["unique_index"]],
    )


def mexico_north(ds):
    required_metrics = ['h25_Nelson', 'h90_Nelson', 'acq3_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    conifer_ind = 0
    return (
        -3.364 * ds['h25_Nelson']
        + 3.210 * ds['h90_Nelson']
        + 21.612 * ds['acq3_Nelson']
        + 18.778 * conifer_ind
    )


def mexico_north_conifer(ds):
    required_metrics = ['h25_Nelson', 'h90_Nelson', 'acq3_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    conifer_ind = 1
    return (
        -3.364 * ds['h25_Nelson']
        + 3.210 * ds['h90_Nelson']
        + 21.612 * ds['acq3_Nelson']
        + 18.778 * conifer_ind
    )


def mexico_south(ds):
    required_metrics = ['h75_Nelson', 'h10_Nelson', 'trail_Nelson']
    ds = get_all_height_metrics(ds, required_metrics)
    return 6.845 * ds['h75_Nelson'] - 6.144 * ds['h10_Nelson'] - 2.565 * ds['trail_Nelson']


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
    ln_biomass = 2.10 + 1.56 * np.log(ds['QMCH']) + 0.05 * np.log(ds['Height_35_to_40'])
    return np.exp(ln_biomass)


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


def western_boreal_eurasia_confier(ds):
    required_metrics = ['h90_Neigh', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 5.95 * ds['h90_Neigh'] - 5 * ds['h25_Neigh'] + 4.72


def western_boreal_eurasia_mixed_hard_wood(ds):
    required_metrics = ['h75_Neigh', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 8.84 * ds['h75_Neigh'] - 5.09 * ds['h25_Neigh'] - 7.03


def eastern_boreal_eurasia(ds):
    required_metrics = ['h75_Neigh', 'h25_Neigh']
    ds = get_all_height_metrics(ds, required_metrics)
    return 13.60 * ds['h75_Neigh'] - 14.30 * ds['h25_Neigh'] - 3.49


def palearctic_wang_2013(ds):
    required_metrics = ['VH']
    ds = get_all_height_metrics(ds, required_metrics)
    return np.exp(1.48 + 1.09 * np.log(ds['VH']))


def palearctic_takagi_2015(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (6.70 * ds['MeanH'] + 24.5)


def palearctic_yavasli_2016(ds):
    required_metrics = ['HOME_Yavasli', 'pct_HOME_Yavasli']
    ds = get_all_height_metrics(ds, required_metrics)
    return 24.11 - 3.84 * ds['HOME_Yavasli'] + 2.44 * ds['pct_HOME_Yavasli']


def palearctic_brovkina_2015(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 12.8 * ds['MeanH'] - 99.9


def palearctic_alberti_2013(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (3.33 * np.power(ds['MeanH'], 1.27))


def palearctic_whrc(ds):
    if 'modeled_wf' not in ds:
        ds['modeled_wf'] = get_modeled_waveform(ds)
    required_metrics = ['HOME_Baccini', 'CANOPY_DEP', 'VH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 13.7949 + 0.8912 * ds['HOME_Baccini'] + 25.4467 * ds['CANOPY_DEP'] - 18.1995 * ds['VH']


def palearctic_shang_and_chazette_2014(ds):
    required_metrics = ['QMCH']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (42.36 + 0.24 * np.square(ds['QMCH']))


def palearctic_simonson_2016(ds):
    required_metrics = ['MeanH']
    ds = get_all_height_metrics(ds, required_metrics)
    return np.exp(3.02 + 0.89 * np.log(ds['MeanH']))


def palearctic_patenaude_2004(ds):
    # if the allometric equation is based on Patenaude et al 2004, we limited the input data to the range for which the equation was fitted
    # by dropping any out of range shots in post-process biomass
    # this procedure is implemented following Mary Farina's manuscript, and the range (13-25m) is based on Figure 4a of Patenaude
    # "Quantifying forest above ground carbon content using LiDAR remote sensing"
    required_metrics = ['h80_p12']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (1.36 * 0.68 * 0.49) * 19.186 * np.exp(0.1256 * ds['h80_p12'])


def suganuma_2006(ds):
    required_metrics = ['treecover2000_mean']
    ds = get_all_height_metrics(ds, required_metrics)
    return 119.9699 * np.power(ds['treecover2000_mean'] / 100.0, 1.1781)


def australia_beets_2011(ds):
    required_metrics = ['h30_canopy', 'treecover2000_mean']
    ds = get_all_height_metrics(ds, required_metrics)
    return 2 * (-45.8 + 7.52 * ds['h30_canopy'] + 0.67 * ds['treecover2000_mean'])


def australia_lucas_2008(ds):
    required_metrics = [
        'h05_canopy',
        'h10_canopy',
        'h20_canopy',
        'h50_canopy',
        'h75_canopy',
        'h80_canopy',
        'treecover2000_mean',
    ]
    ds = get_all_height_metrics(ds, required_metrics)
    return (
        -44.4 * ds['h05_canopy']
        + 57.98 * ds['h10_canopy']
        - 18.8 * ds['h20_canopy']
        + 8.3 * ds['h50_canopy']
        - 34.98 * ds['h75_canopy']
        + 32.2 * ds['h80_canopy']
        + 0.86 * ds['treecover2000_mean']
        - 20.68
    )


ALLOMETRIC_EQUATIONS_MAP = {
    'afrotropic': tropics,
    'tropical_asia': tropics,
    'tropical_neotropic': tropics,
    'extratropical_neotropic': extratropical_neotropic,
    'ak_wetland_steep': ak_wetland_steep,
    'ak_wetland_flat': boreal_wetland,
    'ak_hardwood': w_canada_hardwood,
    'ak_conifer': ak_conifer,
    'ak_mixedwood': ak_mixedwood,
    'ak_burned': w_canada_burned,
    'w_canada_wetland': boreal_wetland,
    'w_canada_hardwood': w_canada_hardwood,
    'w_canada_conifer': w_canada_conifer,
    'w_canada_mixedwood': canada_mixedwood,
    'w_canada_burned': w_canada_burned,
    'e_canada_wetland_steep': e_canada_wetland_steep,
    'e_canada_wetland_flat': boreal_wetland,
    'e_canada_hardwood': e_canada_hardwood,
    'e_canada_conifer': e_canada_conifer,
    'e_canada_mixedwood_high': e_canada_mixedwood_high,
    'e_canada_mixedwood_low': canada_mixedwood,
    'e_canada_burned': e_canada_burned,
    'conus_wetland': conus_wetland,
    'conus_hardwood': conus_hardwood,
    'conus_mixedwood': conus_mixedwood,
    'conus_burned': conus_burned,
    'zero_biomass': zero_biomass,
    'mexico_north': mexico_north,
    'mexico_north_conifer': mexico_north_conifer,
    'mexico_south': mexico_south,
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
    'wbe_confier': western_boreal_eurasia_confier,
    'wbe_mixed_hard_wood': western_boreal_eurasia_mixed_hard_wood,
    'eastern_boreal_eurasia': eastern_boreal_eurasia,
    'palearctic_wang_2013': palearctic_wang_2013,
    'palearctic_takagi_2015': palearctic_takagi_2015,
    'palearctic_yavasli_2016': palearctic_yavasli_2016,
    'palearctic_brovkina_2015': palearctic_brovkina_2015,
    'palearctic_alberti_2013': palearctic_alberti_2013,
    'palearctic_whrc': palearctic_whrc,
    'palearctic_shang_and_chazette_2014': palearctic_shang_and_chazette_2014,
    'palearctic_simonson_2016': palearctic_simonson_2016,
    'palearctic_patenaude_2004': palearctic_patenaude_2004,
    'palearctic_suganuma_2006': suganuma_2006,
    'australia_beets_2011': australia_beets_2011,
    'australia_suganuma_2006': suganuma_2006,
    'australia_lucas_2008': australia_lucas_2008,
    'australia_baccini_2012': tropics,
}


def assign_ecoregion_values(data, tiles):
    """
    Given a dataset with lat, lon, assign the correct ecoregions values for each record
    """
    # error out if lat lon are not in dataset
    for v in ['lat', 'lon']:
        if v not in data:
            raise KeyError(f'required variable {v} not found in input data')

    # load ecoregions map
    # ecoregions map contains Ecoregions2017, NLCD, and EOSD data
    ecoregions_mask = utils.open_ecoregion_data(tiles)
    eco_records = utils.find_matching_records(data=ecoregions_mask, lats=data.lat, lons=data.lon)
    for v in eco_records:
        data[v] = eco_records[v]
    del ecoregions_mask

    # NLCD and EOSD data only cover parts of the globe, set the rest to nan
    for metric in ['nlcd', 'eosd']:
        if metric not in data:
            data[metric] = xr.DataArray(
                np.nan,
                dims=["unique_index"],
                coords=[data['lat'].coords["unique_index"]],
            )

    # 0 and 241 are both null values for nlcd
    data['nlcd'] = xr.where(data.nlcd.isin([0, 241]), x=np.nan, y=data.nlcd)

    # read IGBP data and assign those values to each record as well
    igbp = utils.open_igbp_data(tiles)
    igbp_records = utils.find_matching_records(
        data=igbp, lats=data.lat, lons=data.lon, years=data.datetime.dt.year
    )
    data['igbp'] = igbp_records.igbp
    del igbp

    return data


def assign_treecover_values(data, tiles):
    hansen = []
    for tile in tiles:
        lat, lon = utils.get_lat_lon_tags_from_tile_path(tile)
        # get Hansen data
        hansen_tile = cat.hansen_change(variable='treecover2000', lat=lat, lon=lon).to_dask()
        hansen_tile = hansen_tile.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
        hansen.append(hansen_tile.to_dataset(name='treecover2000', promote_attrs=True))
    hansen = xr.combine_by_coords(hansen, combine_attrs="drop_conflicts").chunk(
        {'lat': 2000, 'lon': 2000}
    )

    hansen_records = utils.find_matching_records(data=hansen, lats=data.lat, lons=data.lon)
    data['treecover2000_mean'] = hansen_records['treecover2000']

    del hansen

    return data


def assign_burned_area_values(data, tiles):
    burned_area = utils.open_burned_area_data(tiles)
    burned_area_rec = utils.find_matching_records(data=burned_area, lats=data.lat, lons=data.lon)

    # if the GLAS shot is burned after year 2000 but prior to year of GLAS acquisition
    # set "burned" to True, else to False
    year_burned = np.floor(burned_area_rec.burned_date / 1000.0)
    data['burned'] = data.datetime.dt.year < year_burned

    del burned_area

    return data


def assign_allometric_eq(ds):
    """
    Given a dataset with ecoregion values (Ecoregions2017, NLCD, EOSD, IGBP), f_slope, and senergy, assign the correct allometric equation
    """
    # calculate f_slope and senergy if not in dataset
    for v in ['f_slope', 'senergy']:
        if v not in ds:
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
        (ds.nlcd == 90) | (ds.eosd == 81) | (ds.nlcd.isnull() & ds.eosd.isnull() & (ds.igbp == 11))
    )
    hardwood_mask = (
        (ds.nlcd == 41)
        | ds.eosd.isin([220, 221, 222, 223])
        | (ds.nlcd.isnull() & ds.eosd.isnull() & (ds.igbp == 4))
    )
    conifer_mask = (
        (ds.nlcd == 42)
        | ds.eosd.isin([210, 211, 212, 213])
        | (ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([1, 3]))
    )
    mixedwood_mask = (
        (ds.nlcd == 43)
        | ds.eosd.isin([51, 230, 231, 232, 233])
        | (ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([5, 8, 9]))
    )
    burned_mask = ds.burned
    zero_biomass_mask = (
        ds.nlcd.isin([11, 12, 21, 22, 23, 24, 31, 52, 71, 72, 73, 74, 81, 82, 95])
        | ds.eosd.isin([0, 10, 11, 12, 20, 30, 31, 32, 33, 34, 35, 36, 37, 40, 51, 52, 82, 83, 100])
        | (ds.nlcd.isnull() & ds.eosd.isnull() & ds.igbp.isin([0, 6, 7, 10, 12, 13, 15, 16]))
    )

    # process boreal nearctic
    alaska_mask = output == 'alaska'
    output = xr.where(
        alaska_mask & wetland_mask & (ds.f_slope >= 2.552), x='ak_wetland_steep', y=output
    )
    output = xr.where(
        alaska_mask & wetland_mask & (ds.f_slope < 2.552), x='ak_wetland_flat', y=output
    )
    output = xr.where(alaska_mask & hardwood_mask, x='ak_hardwood', y=output)
    output = xr.where(alaska_mask & conifer_mask, x='ak_conifer', y=output)
    output = xr.where(alaska_mask & mixedwood_mask, x='ak_mixedwood', y=output)
    output = xr.where(alaska_mask & burned_mask, x='ak_burned', y=output)
    output = xr.where(alaska_mask & zero_biomass_mask, x='zero_biomass', y=output)

    western_canada_mask = output == 'western_canada'
    output = xr.where(western_canada_mask & wetland_mask, x='w_canada_wetland', y=output)
    output = xr.where(western_canada_mask & hardwood_mask, x='w_canada_hardwood', y=output)
    output = xr.where(western_canada_mask & conifer_mask, x='w_canada_conifer', y=output)
    output = xr.where(western_canada_mask & mixedwood_mask, x='w_canada_mixedwood', y=output)
    output = xr.where(western_canada_mask & burned_mask, x='w_canada_burned', y=output)
    output = xr.where(western_canada_mask & zero_biomass_mask, x='zero_biomass', y=output)

    eastern_canada_mask = output == 'eastern_canada'
    output = xr.where(
        eastern_canada_mask & wetland_mask & (ds.f_slope >= 2.552),
        x='e_canada_wetland_steep',
        y=output,
    )
    output = xr.where(
        eastern_canada_mask & wetland_mask & (ds.f_slope < 2.552),
        x='e_canada_wetland_flat',
        y=output,
    )
    output = xr.where(eastern_canada_mask & hardwood_mask, x='e_canada_hardwood', y=output)
    output = xr.where(eastern_canada_mask & conifer_mask, x='e_canada_conifer', y=output)
    output = xr.where(
        eastern_canada_mask & mixedwood_mask & (ds.senergy >= 26.643),
        x='e_canada_mixedwood_high',
        y=output,
    )
    output = xr.where(
        eastern_canada_mask & mixedwood_mask & (ds.senergy < 26.643),
        x='e_canada_mixedwood_low',
        y=output,
    )
    output = xr.where(eastern_canada_mask & burned_mask, x='e_canada_burned', y=output)
    output = xr.where(eastern_canada_mask & zero_biomass_mask, x='zero_biomass', y=output)

    # process conus ecoregions
    conus_mask = output == 'conus'
    output = xr.where(conus_mask & wetland_mask, x='conus_wetland', y=output)
    output = xr.where(conus_mask & hardwood_mask, x='conus_hardwood', y=output)
    output = xr.where(conus_mask & mixedwood_mask, x='conus_mixedwood', y=output)
    output = xr.where(conus_mask & burned_mask, x='conus_burned', y=output)
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

    # split mexico_north into two equations based on confifer mask
    mexico_north_mask = output == 'mexico_north'
    output = xr.where(mexico_north_mask & conifer_mask, x='mexico_north_conifer', y=output)

    return output


def filter_on_time_of_year(ds):
    """
    From Farina et al 2018:
    For the regions north of the tropics, we used GLAS laser campaigns L2A, L2C, L3A, L3C, L3D, L3F, L3I, and L3K.
    For regions south of the tropics, we used GLAS laser campaigns L1A, L2B, L2D, L3B, L3E, L3G, L3H, and L3J.
    All of the sixteen campaigns listed above were used for tropical regions.
    When applying the height-biomass equations published by Popescu et al. (2011) and Shang and Chazette (2014),
    we used GLAS data acquired during leaf-off conditions because leaf-off data were used to train the equations.

    Timing of campaigns retrieved from https://icesat.gsfc.nasa.gov/icesat/missionevents.php, https://nsidc.org/data/icesat/orbit_grnd_trck.html
    """
    d0 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    if ds.time.isnull().sum() > 0:
        print('filling nulls')
        filler = ds.datetime.fillna(d0).astype(int) / 1e9
        ds['time'] = ds.time.fillna(filler)

    print(f'{ds.dims["unique_index"]} records before filtering based on time')
    # filter out campaigns 2E and 2F according to Farina et al 2018 (data after mar 1st, 2019)
    d = datetime(2009, 3, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    ds = ds.where(ds.time < d, drop=True)
    # filter out campaign 1B 2003-03-20	2003-03-29
    L1B_start = datetime(2003, 3, 20, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    L1B_end = datetime(2003, 3, 29, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    ds = ds.where(((ds.time < L1B_start) | (ds.time > L1B_end)), drop=True)
    print(f'{ds.dims["unique_index"]} records aftering filtering out campaign L2E, L2F, L1B')

    # define regions north or south of the tropics
    special_eq = ds.allometric_eq.isin(
        ['conus_conifer_popescu_2011', 'palearctic_shang_and_chazette_2014']
    )
    north = ds.lat > 23.5
    south = ds.lat < -23.5

    # define time
    # 2006-10-25 to 2006-11-27
    L3G_start = datetime(2006, 10, 25, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    L3G_end = datetime(2006, 11, 28, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    # 2008-11-25 to 2008-12-17
    L2D_start = datetime(2008, 11, 25, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    L2D_end = datetime(2008, 12, 18, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    north_leaf_on = (
        (ds.datetime.dt.month >= 5)
        & ((ds.time < L3G_start) | (ds.time > L3G_end))
        & ((ds.time < L2D_start) | (ds.time > L2D_end))
    )
    south_leaf_on = (
        (ds.datetime.dt.month < 5)
        | ((ds.time >= L3G_start) & (ds.time <= L3G_end))
        | ((ds.time >= L2D_start) & (ds.time <= L2D_end))
    )
    # if a record is: 1) in the north, 2) not in north leaf on time period, and 3) not in special eq, drop them
    ds = ds.where(~(north & ~north_leaf_on & ~special_eq), drop=True)
    ds = ds.where(~(south & ~south_leaf_on & ~special_eq), drop=True)

    # if a record in: 1) in the special eq, 2) in the north, 3) not in the north leaf off (aka south leaf on) time period, drop them
    ds = ds.where(~(north & ~south_leaf_on & special_eq), drop=True)
    ds = ds.where(~(south & ~north_leaf_on & special_eq), drop=True)
    print(f'{ds.dims["unique_index"]} records aftering filtering based on leaf on/off conditions')

    return ds


def calculate_biomass(ds):
    variables = list(ds.keys()) + ['biomass']

    out = []
    missing_eq_name = []
    for eq_name, group in ds.groupby('allometric_eq'):
        if eq_name in ALLOMETRIC_EQUATIONS_MAP:
            biomass = ALLOMETRIC_EQUATIONS_MAP[eq_name](group)
            group['biomass'] = biomass
            out.append(group[variables])
        else:
            missing_eq_name.append(eq_name)

    # if len(missing_eq_name) > 0:
    #     print(
    #         f'Dropping {ds.dims["unique_index"] - out.dims["unique_index"]} records out of {ds.dims["unique_index"]} due to missing equations {missing_eq_name}'
    #     )
    return xr.concat(out, dim='unique_index')


def post_process_biomass(ds):
    ds = get_all_height_metrics(ds, metrics=['VH', 'treecover2000_mean', 'h80_p12'])
    # drop records where VH or treecover2000_mean are negative
    mask = (ds.VH >= 0) & (ds.treecover2000_mean >= 0)
    ds = ds.where(mask, drop=True)

    # if the allometric equation is based on Patenaude et al 2004, we limited the input data to the range for which the equation was fitted
    # this procedure is implemented following Mary Farina's manuscript, and the range (13-25m) is based on Figure 4a of Patenaude
    # Quantifying forest above ground carbon content using LiDAR remote sensing
    to_drop = (ds.allometric_eq == 'palearctic_patenaude_2004') & (
        (ds.h80_p12 > 25) | (ds.h80_p12 < 13)
    )
    ds = ds.where(~to_drop, drop=True)

    # From Farina 2018: if VH < 2, predicted biomass < 1 Mg/ha, or canopy cover < 10%, set biomass to 0
    mask = (ds.VH < 2) | (ds.biomass < 1) | (ds.treecover2000_mean < 10)
    # print(
    #     f'Setting the biomass to 0 for {mask.mean().values * 100}% of records based on Harris et al procedures'
    # )
    ds['biomass'] = xr.where(mask, x=0, y=ds.biomass)

    return ds


def apply_allometric_equation(ds, min_lat, max_lat, min_lon, max_lon):
    """
    Given a dataset containing GLAS records, assign allometric equation based on appropriate info,
    then apply the allometric equation with respective height metrics to get biomass
    """

    # find a list of 10x10 degree tile names covering the bounding box
    # the ancillary data used in preprocess are stored as these 10x10 degree tiles
    tiles = utils.find_tiles_for_bounding_box(min_lat, max_lat, min_lon, max_lon)

    # assign ecoregions and canopy cover data
    ds = assign_ecoregion_values(ds, tiles)
    ds = assign_treecover_values(ds, tiles)
    ds = assign_burned_area_values(ds, tiles)

    # assign allometric eq
    ds["allometric_eq"] = assign_allometric_eq(ds)
    ds = filter_on_time_of_year(ds)

    # apply allometric equations
    ds = calculate_biomass(ds)
    ds = post_process_biomass(ds)

    return ds


def get_realm_from_ecoregion(ecoregions):
    """
    input and output are both data arrays
    """
    ECO_TO_REALM_MAP[-999] = np.nan

    realms = xr.apply_ufunc(
        ECO_TO_REALM_MAP.__getitem__,
        ecoregions.fillna(-999),
        vectorize=True,
        dask='parallelized',
        output_dtypes=['object'],
    )

    return realms.astype(str)
