<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# carbonplan / trace

**working repo for carbonplan's climate trace project**

[![GitHub][github-badge]][github]
[![Build Status]][actions]
![MIT License][]
[![Binder](https://aws-uswest2-binder.pangeo.io/badge_logo.svg)](https://aws-uswest2-binder.pangeo.io/v2/gh/carbonplan/trace/main?filepath=notebooks%2Fblogpost_sample_notebook.ipynb)

[github]: https://github.com/carbonplan/trace
[github-badge]: https://badgen.net/badge/-/github?icon=github&label
[build status]: https://github.com/carbonplan/trace/actions/workflows/main.yaml/badge.svg
[actions]: https://github.com/carbonplan/trace/actions/workflows/main.yaml
[mit license]: https://badgen.net/badge/license/MIT/blue

This repository includes example Jupyter notebooks and other utilities for a collaborative project CarbonPlan is working on involving tracking emissions related to deforestation and other land use changes.

This project is just getting going. Nothing here is final or complete.

We are in the process of incorporating the scripts and notebooks which were used to implement version 0 of the dataset. Some tips for reproducing this effort:

* the `scripts/aggregate_emissions.v0.py` script can be run to reproduce both the 3 km global emissions raster dataset and the country-average estimates. As a warning, depending on the size of the machine you're running on, you might encounter memory issues when dealing with the 30m datasets. For that reason, we opted to process the 30m tiles in serial. If you are struggling, you might want to check to ensure that your cluster isn't getting overloaded.
* currently country estimates are based upon geopolitical boundaries as specified by Natural Earth (https://www.naturalearthdata.com/) as countries are defined in 2020 (meaning that geopolitical boundaries that have changed over the period of record will be tagged to a static country designation as defined in 2020).

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed. When possible, the data used by this project is licensed using the [CC-BY-4.0](https://choosealicense.com/licenses/cc-by-4.0/) license. We include attribution and additional license information for third party datasets, and we request that you also maintain that attribution if using this data.

## about us

CarbonPlan is a non-profit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/trace/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
