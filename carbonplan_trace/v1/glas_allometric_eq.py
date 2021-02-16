def tsui_etal_2012(ds):
    return (
        -7.144
        - 12.925 * ds["meanH_height"]
        + 2.239 * ds["10th_height"]
        + 14.019 * ds["90th_height"]
    )


def apply_allometric_equation(ds):
    ds["biomass"] = tsui_etal_2012(ds)

    return ds


ALLOMETRIC_EQUATIONS_MAP = {}
