features = (
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'NDII']
    + [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
    + ['burned', 'treecover2000_mean']
    + ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']
    + ['elev', 'slope', 'aspect']
)


def calc_NDII(df):
    nir = df['SR_B4']
    swir = df['SR_B5']
    return (nir - swir) / (nir + swir)


def calc_NDVI(df):
    nir = df['SR_B4']
    red = df['SR_B3']
    return (nir - red) / (nir + red)
