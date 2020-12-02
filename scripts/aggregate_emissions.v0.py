import fsspec
import numcodecs
import numpy as np
import tqdm
import xarray as xr

TC02_PER_TC = 3.67
TC_PER_TBM = 0.5
SQM_PER_HA = 10000
ORNL_SCALING = 0.1


def compute_grid_area(da):

    """
    Compute the geographic area (in square meters) of every pixel in the data array provided.

    Parameters
    ----------
    da : xarray data array
        Data array with spatial coordinates lon and lat

    Returns
    -------
    areacella / SQM_PER_HA : data array
        Data array with grid cell areas in square meters
    """

    R = 6.371e6
    dϕ = np.radians((da["lat"][1] - da["lat"][0]).values)
    dλ = np.radians((da["lon"][1] - da["lon"][0]).values)
    dA = R ** 2 * np.abs(dϕ * dλ) * np.cos(np.radians(da["lat"]))
    areacella = dA * (0 * da + 1)

    return areacella / SQM_PER_HA


with fsspec.open(
    "https://storage.googleapis.com/earthenginepartners" "-hansen/GFC-2018-v1.6/treecover2000.txt"
) as f:
    lines = f.read().decode().splitlines()
print("We are working with {} different files".format(len(lines)))


lat_tags = [
    "80N",
    "70N",
    "60N",
    "50N",
    "40N",
    "30N",
    "20N",
    "10N",
    "00N",
    "10S",
    "20S",
    "30S",
    "40S",
    "50S",
]
lon_tags = [f"{n:03}W" for n in np.arange(10, 190, 10)] + [
    f"{n:03}E" for n in np.arange(0, 190, 10)
]

# the arrays where you'll throw your active lat/lon permutations
lats = []
lons = []

encoding = {"emissions": {"compressor": numcodecs.Blosc()}}

for line in lines:
    pieces = line.split("_")
    lat = pieces[-2]
    lon = pieces[-1].split(".")[0]

    if (lat in lat_tags) and (lon in lon_tags):
        lats.append(lat)
        lons.append(lon)

list_all_coarsened = []
for lat, lon in tqdm(list(zip(lats, lons))):
    try:
        # We only have data over land so this will throw
        # an exception if
        mapper = fsspec.get_mapper(
            f"gs://carbonplan-scratch/global-forest-emissions/{lat}_{lon}.zarr"
        )
        da_global = xr.open_zarr(mapper)
        da_mask = da_global.isel(year=0, drop=True)
        da_area = compute_grid_area(da_mask)
        list_all_coarsened.append((da_global * da_area).coarsen(lat=100, lon=100).sum().compute())
    except ValueError:
        print("{} {} didnt work (likely because its ocean) booooo".format(lat, lon))

coarsened_url = "gs://carbonplan-scratch/global-forest-emissions/global/3000m/raster.zarr"

mapper = fsspec.get_mapper(coarsened_url)

combined_ds = xr.combine_by_coords(list_all_coarsened, compat="override", coords="minimal")
combined_ds.to_zarr(mapper, encoding=encoding, mode="w")
