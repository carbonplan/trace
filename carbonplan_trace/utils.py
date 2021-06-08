def zarr_is_complete(store, check='.zmetadata'):
    return check in store
