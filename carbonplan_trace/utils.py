import importlib
import os


def get_versions(
    packages=[
        'carbonplan',
        'carbonplan_trace',
        'xarray',
        'dask',
        'numpy',
        'scipy',
        'fsspec',
        'intake',
        'rasterio',
        'zarr',
    ]
):
    """Helper to fetch commonly used package versions

    Parameters
    ----------
    packages : list
        List of packages to fetch versions for

    Returns
    -------
    versions : dict
        Version dictionary with keys of package names and values of version strings
    """
    versions = {'docker_image ': os.getenv('REPO_HASH', None)}

    for p in packages:
        try:
            mod = importlib.import_module(p)
            versions[p] = getattr(mod, '__version__', None)
        except ModuleNotFoundError:
            versions[p] = None

    return versions


def zarr_is_complete(store, check='.zmetadata'):
    """Return true if Zarr store is complete"""
    if not isinstance(check, list):
        check = [check]

    out = True
    for c in check:
        out = out & (c in store)
    return out
