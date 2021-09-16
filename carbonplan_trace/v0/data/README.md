# V0 Data

The primary datasets used and developed for our V0 dataset are listed in the [this](./catalog.yaml) [Intake catalog](https://intake.readthedocs.io/en/latest/).

If you are using the `carbonplan_trace` Python package, you can access the catalog via:

```python
from carbonplan_trace.v0.data import cat

emissions_tile = cat.emissions_30m(lat='00N', lon='060W').to_dask()  # returns an xarray.Dataset
```

The same catalog object can be loaded directly using intake:

```python
import intake

cat = intake.open_catalog(
    'https://raw.githubusercontent.com/carbonplan/trace/main/carbonplan_trace/v0/data/catalog.yaml'
)
```

## Input datasets

Here we highlight the primary input datasets used in V0:

| Dataset         | Description                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hansen_change` | [Global 30m treecover and loss/gain from Hansen et al. (2013)](https://storage.cloud.google.com/earthenginepartners-hansen/GFC-2020-v1.8/download.html) |
| `gfw_biomass`   | [Global 30m aboveground live woody biomass density](https://data.globalforestwatch.org/datasets/gfw::aboveground-live-woody-biomass-density/about)      |

## Data products

Here we highlight the main data products produced in V0:

| Dataset                     | Description                                                                       |
| --------------------------- | --------------------------------------------------------------------------------- |
| `emissions_30m`             | Global 30 m gross emissions                                                       |
| `emissions_30m_partitioned` | Global 30 m gross emissions partitioned into fire and non-fire related components  |
| `emissions_3km`             | Global 3 km gross emissions                                                       |
| `emissions_3km_partitioned` | Global 3 km gross emissions partitioned into fire and non-fire related components |
