import intake
import xarray as xr

cat = intake.open_catalog("catalog.yaml")


def _preprocess(da, lat=None, lon=None):
    da = da.rename({"x": "lon", "y": "lat"}).squeeze(drop=True)
    if lat is not None:
        da = da.assign_coords(lat=lat, lon=lon)
    return da


def open_hansen_2018_tile(lat, lon):
    ds = xr.Dataset()

    # Min Hansen data
    variables = ["treecover2000", "gain", "lossyear", "datamask", "first", "last"]
    for v in variables:
        da = cat.hansen_2018(variable=v, lat=lat, lon=lon).to_dask().pipe(_preprocess)
        # force coords to be identical
        if ds:
            da = da.assign_coords(lat=ds.lat, lon=ds.lon)
        ds[v] = da

    ds["treecover2000"] /= 100.0
    ds["lossyear"] += 2000

    # Hansen biomass
    ds["agb"] = (
        cat.hansen_biomass(lat=lat, lon=lon).to_dask().pipe(_preprocess, lat=ds.lat, lon=ds.lon)
    )

    # Hansen emissions
    ds["emissions_ha"] = (
        cat.hansen_emissions_ha(lat=lat, lon=lon)
        .to_dask()
        .pipe(_preprocess, lat=ds.lat, lon=ds.lon)
    )
    ds["emissions_px"] = (
        cat.hansen_emissions_px(lat=lat, lon=lon)
        .to_dask()
        .pipe(_preprocess, lat=ds.lat, lon=ds.lon)
    )

    return ds
